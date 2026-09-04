import argparse
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from pathlib import Path

from deepeval.test_case import LLMTestCase

from tool.config_reader import ConfigReader
from tool.concurrency import ProgressEvent, emit_terminal_progress, run_in_thread_pool
from tool.csv_writer import CsvWriter
from tool.file_utils import mkdir_with_timestamp
from tool import AsyncResultWriter, MarkdownWriter
from tool.collection_result import CollectionResult
from tool.get_bad_cases import extract_bad_cases
from tool.log_factory import LogFactory
from agents.factory import create_agent, get_enabled_classes, is_agent_only_enabled
from pipeline.test_case_loader import make_test_case_list, make_tmp_test_case_list
from pipeline.resume import prepare_resume
from pipeline.placeholder_filler import fill_test_cases, fill_raw_datasets, preview_fill
from evaluator.runner import run_evaluate, run_evaluate_structured, run_multimodel_reevaluate, recompute_overall_success

logger = LogFactory.get_logger(__name__)

# 检索类指标名称集合 — 用例 metrics 中包含任一即触发 RAGFlow 检索轨
RETRIEVAL_METRIC_NAMES = {'mrr', 'recallk', 'precisionk'}

#CLI
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="执行评测套件")
    parser.add_argument(
        '-cp',
        '--csv_path',
        type=str,
        help='可以指定测试数据集csv文件路径，需要指定文件名，'
             '不指定默认执行test_suite下所有csv格式的测试用例集。'
    )
    parser.add_argument(
        '-m',
        '--metrics',
        choices=['reverse_validation', 'contextual_recall'],
        nargs='+',
        help='指定评测指标，默认为reverse_validation，'
             '可以选择reverse_validation和contextual_recall两种指标，'
             '单独使用或同时使用两者进行评测。指定测试数据集路径时该参数必传'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='从tmp目录恢复评测，跳过Agent调用阶段，直接从组装llm_test_case处重入'
    )
    parser.add_argument(
        '--resume-result-dir',
        help='可选的历史 result/<时间戳> 目录；指定则校验 checkpoint 并复用通过结果，'
             '不指定则从 tmp 全量重新评价',
    )
    parser.add_argument(
        '-a', '--agent_classes',
        nargs='+',
        help='指定调用的 agent 类名，支持多个，如: -a Diagnosis 或 -a PVAssistant Diagnosis'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='占位符填充随机种子，不传则随机生成（生成的 seed 会输出到日志和评测报告）'
    )
    parser.add_argument(
        '--fill-preview',
        action='store_true',
        help='仅执行占位符填充预览，输出到 placeholder_fill.output_dir 后退出，不执行评测'
    )
    parser.add_argument(
        '--refresh-entities',
        action='store_true',
        help='填充前先调用业务平台接口更新 entity_mapping.json（登录/接口失败时中断执行）'
    )
    args = parser.parse_args()
    if args.resume_result_dir is not None and not args.resume:
        parser.error('--resume-result-dir 必须与 --resume 一起使用')
    if args.resume and args.fill_preview:
        parser.error('--resume 不能与 --fill-preview 一起使用')
    return args


def _resolve_tenant_id(source_csv: str) -> str | None:
    """从测试用例源文件路径提取后缀，查找对应的 X-Tenant-Id。

    文件命名规范: test_cases_<suffix>.csv
    例如 test_cases_normal.csv → suffix=normal → 查 tenant_id_map

    :param source_csv: 测试用例 CSV 文件路径（即 _source_csv 字段值）
    :return: 匹配到的 tenant_id 字符串，未匹配返回 None
    """
    if not source_csv:
        return None
    filename = os.path.basename(source_csv)
    if not (filename.startswith('test_cases_') and filename.endswith('.csv')):
        return None
    suffix = filename[len('test_cases_'):-len('.csv')]
    conf = ConfigReader.get_instance()
    tenant_map = conf.get('agents.http_agent.class_config.PVAssistant.tenant_id_map', {})
    if suffix in tenant_map:
        return tenant_map[suffix]
    return tenant_map.get('default', None)


def _case_progress_label(test_case: dict) -> str:
    """提取适合终端展示的短用例标识。"""
    for key in ('用例编号', 'case_id', 'id', 'query'):
        value = test_case.get(key)
        if value is not None and str(value).strip():
            return ' '.join(str(value).split())[:80]
    return '?'


def _emit_retry_progress(task_name: str, test_case: dict, attempt: int,
                         max_retries: int, retry_delay: float, error: Exception,
                         context: str = '') -> None:
    """输出独立于 logging 配置的 Agent 重试事件。"""
    context_text = f' {context}' if context else ''
    emit_terminal_progress(ProgressEvent(
        event='retry',
        task_name=task_name,
        message=(
            f'用例={_case_progress_label(test_case)}{context_text} '
            f'尝试={attempt + 2}/{max_retries + 1} '
            f'原因={type(error).__name__}: {error} 等待={retry_delay}s'
        ),
    ))


#Agent 调用薄函数
def call_agent(agent, test_case_csv: dict):
    """agent调用入口，内置重试逻辑，成功时将响应并入字典"""
    conf = ConfigReader.get_instance()
    max_retries = conf.get("agents.http_agent.call_agent_max_retries", 3)
    retry_delay = conf.get("agents.http_agent.call_agent_retry_delay", 2)

    query = test_case_csv.get('query', '')
    logger.info(f'call_agent, query: {query[:80]}')

    tenant_id = test_case_csv.get('_tenant_id')
    language = test_case_csv.get('language')
    request_kwargs = {'tenant_id': tenant_id}
    if language is not None and str(language).strip():
        request_kwargs['language'] = str(language).strip()

    for attempt in range(max_retries + 1):
        try:
            answer_raw, answer_summary, res_time = agent.call_agent(query, **request_kwargs)
            test_case_csv['agent_response'] = answer_summary
            test_case_csv['res_time(s)'] = res_time
            logger.debug(answer_raw)
            logger.info(f'call_agent成功, query: {query[:80]}')
            return
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f'call_agent失败，第{attempt + 1}/{max_retries}次重试, '
                    f'query: {query[:80]}, 错误: {e}'
                )
                _emit_retry_progress(
                    'call_agent', test_case_csv, attempt,
                    max_retries, retry_delay, e,
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f'call_agent最终失败（已重试{max_retries}次）, '
                    f'query: {query[:80]}, 错误: {e}'
                )
                raise


def _call_agent_dataqa_direct(agent, test_case_csv: dict):
    """DataQA direct_inquiry 两段式调用: new_query → confirm_execute，内置重试"""
    conf = ConfigReader.get_instance()
    max_retries = conf.get("agents.http_agent.call_agent_max_retries", 3)
    retry_delay = conf.get("agents.http_agent.call_agent_retry_delay", 2)

    # 数据集列的 actual_capability_id 是预期值（如 station_overview），
    # 稍后会被覆盖为 agent 实际返回值，先保存到 expected_capability_id 供断言使用。
    # 仅首次保存：--resume 重入时 tmp 文件已含该列，避免把上一轮的实际值误当预期
    if 'expected_capability_id' not in test_case_csv:
        test_case_csv['expected_capability_id'] = test_case_csv.get('actual_capability_id', '')

    query = test_case_csv.get('query', '')
    logger.info(f'DataQA direct_inquiry, query: {query[:80]}')

    for attempt in range(max_retries + 1):
        try:
            start = time.perf_counter()
            new_query_raw, confirmation_id, session_id = agent.new_query(query)

            # 第一步未提取到 confirmationId（如实体绑定失败）：
            # 原始响应写入独立列并标记跳过后续断言，不执行 confirm_execute，也不重试
            if confirmation_id is None:
                test_case_csv['new_query_fail_raw'] = json.dumps(
                    new_query_raw, ensure_ascii=False
                )
                test_case_csv['skip_assertion'] = True
                test_case_csv['is_success'] = False
                test_case_csv['用例是否通过'] = False
                test_case_csv['evaluate_error'] = (
                    '第一步未提取到confirmationId，跳过结构化断言'
                    '（原始响应见 new_query_fail_raw 列）'
                )
                logger.warning(
                    f'DataQA new_query 未提取到confirmationId, '
                    f'query: {query[:80]}, 已标记跳过断言'
                )
                return

            test_case_csv['new_query_res_raw'] = json.dumps(
                new_query_raw, ensure_ascii=False
            )

            confirm_raw = agent.confirm_execute(confirmation_id, session_id)
            elapsed = time.perf_counter() - start

            test_case_csv['confirm_res_raw'] = json.dumps(
                confirm_raw, ensure_ascii=False
            )
            test_case_csv['agent_response'] = agent._extract_summary(confirm_raw)
            test_case_csv['actual_capability_id'] = agent.extract_capability_id(confirm_raw)
            test_case_csv['actual_parameters'] = json.dumps(
                agent.extract_parameters(confirm_raw), ensure_ascii=False
            )
            test_case_csv['res_time(s)'] = f'{elapsed:.4f}'
            logger.info(f'DataQA direct_inquiry 成功, query: {query[:80]}')
            return
        except Exception as e:
            if attempt < max_retries:
                logger.warning(
                    f'DataQA direct_inquiry 失败，第{attempt + 1}/{max_retries}次重试, '
                    f'query: {query[:80]}, 错误: {e}'
                )
                _emit_retry_progress(
                    'DataQA-direct', test_case_csv, attempt,
                    max_retries, retry_delay, e,
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f'DataQA direct_inquiry 最终失败（已重试{max_retries}次）, '
                    f'query: {query[:80]}, 错误: {e}'
                )
                raise


def call_ragflow_retrieve(retriever, test_case_csv: dict):
    """调 RAGFlow 检索接口，将文档名列表（document_keyword）写回 case dict

    RAGFlow 调用失败时不抛异常，标记错误让后续 make_llm_case_retrieval 跳过。
    """
    query = test_case_csv.get('query', '')
    logger.info(f'RAGFlow检索, query: {query[:80]}')
    try:
        _, chunks, _ = retriever.retrieve(question=query)
        test_case_csv['retrieved_docs'] = [
            _normalize_whitespace(c.get('document_keyword', '')) for c in chunks
        ]
        logger.info(f'RAGFlow检索成功, query: {query[:80]}, 召回{len(chunks)}条')
    except Exception as e:
        logger.error(f'RAGFlow检索失败, query: {query[:80]}, 错误: {e}')


def make_llm_case(test_case_csv: dict):
    """llm_case组装方法，组装完成的用例重新放回字典中

    agent_response 缺失时（call_agent 失败）跳过该用例，写入错误标记。
    """
    if _is_skip_assertion(test_case_csv):
        logger.debug(
            f"[跳过] query='{test_case_csv.get('query', '?')[:60]}' "
            f"已标记 skip_assertion，不组装 LLMTestCase"
        )
        return

    if 'agent_response' not in test_case_csv:
        logger.debug(
            f"[跳过] query='{test_case_csv.get('query', '?')[:60]}' "
            f"agent_response 缺失，无法组装 LLMTestCase"
        )
        test_case_csv['is_success'] = False
        test_case_csv['evaluate_error'] = 'Agent调用失败，无agent_response'
        return

    llm_test_case = LLMTestCase(
        input=test_case_csv['query'],
        expected_output=test_case_csv.get('expected_behavior', ''),
        context=[test_case_csv.get('forbidden_behavior', '')],
        retrieval_context=[test_case_csv['agent_response']]
    )
    test_case_csv['llm_test_case'] = llm_test_case


def make_llm_case_structured(test_case_csv: dict):
    """为 DataQA 用例组装结构化断言的 LLMTestCase。

    capability_id 和 parameters 各自独立的 LLMTestCase：
      - capability: expected_output=预期字符串, retrieval_context=[实际字符串]，纯 == 比对
      - parameters:  expected_output=预期 JSON 字符串, retrieval_context=[实际 JSON 字符串]，子集比对

    actual_capability_id / actual_parameters 任一缺失时标记失败并跳过组装。
    """
    query = test_case_csv.get('query', '')

    if _is_skip_assertion(test_case_csv):
        logger.debug(
            f"[跳过] query='{query[:60]}' 已标记 skip_assertion，跳过结构化断言"
        )
        return

    missing = []

    if 'actual_capability_id' not in test_case_csv:
        missing.append('未能正常捕获 capability_id')
    if 'actual_parameters' not in test_case_csv:
        missing.append('未能正常捕获 parameters')

    if missing:
        test_case_csv['is_success'] = False
        test_case_csv['evaluate_error'] = '；'.join(missing)
        logger.debug(
            f"[跳过] query='{query[:60]}' "
            f"{'；'.join(missing)}，跳过结构化断言"
        )
        return

    test_case_csv['llm_test_case_capability'] = LLMTestCase(
        input=query,
        expected_output=test_case_csv.get('expected_capability_id', ''),
        retrieval_context=[test_case_csv['actual_capability_id']],
    )

    test_case_csv['llm_test_case_params'] = LLMTestCase(
        input=query,
        expected_output=test_case_csv.get('expected_parameters', '{}'),
        retrieval_context=[test_case_csv['actual_parameters']],
    )


def make_llm_case_retrieval(test_case_csv: dict):
    """为检索评测组装 LLMTestCase

    retrieval_context = 实际召回的文档名列表（document_keyword）
    expected_output   = 预处理后的期望文档名（| 分隔），与 document_keyword 对齐

    retrieved_docs 或 expected_docs 缺失时标记失败并跳过组装。
    """
    if 'retrieved_docs' not in test_case_csv:
        logger.debug(
            f"[跳过] query='{test_case_csv.get('query', '?')[:60]}' "
            f"retrieved_docs 缺失，无法组装检索 LLMTestCase"
        )
        test_case_csv['is_success'] = False
        test_case_csv['evaluate_error'] = 'RAGFlow检索失败，无retrieved_docs'
        return

    raw_expected = test_case_csv.get('expected_docs', '')
    if not raw_expected:
        logger.debug(
            f"[跳过] query='{test_case_csv.get('query', '?')[:60]}' "
            f"expected_docs 为空，无法执行检索评测"
        )
        test_case_csv['is_success'] = False
        test_case_csv['evaluate_error'] = '缺少expected_docs列'
        return

    test_case_csv['llm_test_case_retrieval'] = LLMTestCase(
        input=test_case_csv['query'],
        expected_output=_normalize_expected_docs(raw_expected),
        retrieval_context=test_case_csv['retrieved_docs'],
    )


def _normalize_whitespace(raw: str) -> str:
    """去除字符串前后空白，并将内部多余空白（含换行）压缩为单个空格"""
    if not raw:
        return raw
    return ' '.join(raw.split())


def _normalize_expected_docs(raw: str) -> str:
    """预处理 expected_docs：按 | 分隔后对每段做空白规范化，再以 | 拼回"""
    if not raw:
        return raw
    return '|'.join(
        _normalize_whitespace(doc) for doc in raw.split('|')
        if _normalize_whitespace(doc)
    )


# ---- 多轮对话支持 ----

def _is_multi_turn(case_row: dict) -> bool:
    """判断用例是否为多轮对话，兼容CSV字符串和Python bool"""
    val = case_row.get('is_multi_turn', False)
    if isinstance(val, str):
        return val.strip().upper() in ('TRUE', 'YES', '1')
    return bool(val)


def _is_skip_assertion(case_row: dict) -> bool:
    """判断用例是否被标记跳过断言（第一步未提取到confirmationId），兼容CSV字符串和Python bool"""
    val = case_row.get('skip_assertion', False)
    if isinstance(val, str):
        return val.strip().upper() in ('TRUE', 'YES', '1')
    return bool(val)


def _parse_turns(case_row: dict) -> list[tuple[int, str, str]]:
    """从多轮用例行中提取所有对话轮次 [(轮次号, 问题, 预期结果), ...]，按轮次排序"""
    turns = []
    for key in case_row:
        m = re.match(r'^第(\d+)轮$', key)
        if m:
            turn_num = int(m.group(1))
            question = case_row.get(key, '')
            if question and str(question).strip():
                expected_key = f'第{turn_num}轮对话预期结果'
                expected_output = case_row.get(expected_key, '') or ''
                turns.append((turn_num, str(question).strip(), str(expected_output).strip()))
    turns.sort(key=lambda x: x[0])
    return turns


def call_multi_turn_agent(agent, case_row: dict) -> list[dict]:
    """
    处理一条多轮用例：共享 session_id，严格按轮次串行调用 agent，
    每轮返回一条子行 dict，格式对齐单轮用例方便后续复用 make_llm_case / run_evaluate。

    :return: N 条子行列表，每条包含 query, agent_response, expected_behavior, _turn 等
    """
    conf = ConfigReader.get_instance()
    max_retries = conf.get("agents.http_agent.call_agent_max_retries", 3)
    retry_delay = conf.get("agents.http_agent.call_agent_retry_delay", 2)

    session_id = str(uuid.uuid4())
    turns = _parse_turns(case_row)

    if not turns:
        logger.warning(f"[多轮] 未解析到有效轮次，用例编号={case_row.get('用例编号', '?')}")
        return []

    parent_case_id = case_row.get('用例编号', session_id)
    source_csv = case_row.get('_source_csv', '')
    tenant_id = case_row.get('_tenant_id')
    language = case_row.get('language')
    request_kwargs = {
        'session_id': session_id,
        'tenant_id': tenant_id,
    }
    if language is not None and str(language).strip():
        request_kwargs['language'] = str(language).strip()
    logger.info(f'[多轮] 开始, 用例编号={parent_case_id}, 共{len(turns)}轮, session_id={session_id}')

    sub_rows = []
    for turn_num, question, expected_output in turns:
        logger.info(f'[多轮] 用例={parent_case_id}, 第{turn_num}轮, query: {question[:80]}')

        answer_summary = None
        res_time = None
        for attempt in range(max_retries + 1):
            try:
                answer_raw, answer_summary, res_time = agent.call_agent(
                    question, **request_kwargs
                )
                break
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        f'[多轮] 用例={parent_case_id}, 第{turn_num}轮失败, '
                        f'第{attempt + 1}/{max_retries}次重试, 错误: {e}'
                    )
                    time.sleep(retry_delay)
                else:
                    logger.error(
                        f'[多轮] 用例={parent_case_id}, 第{turn_num}轮最终失败'
                        f'（已重试{max_retries}次）, 错误: {e}'
                    )

        # 继承父用例的全部字段（含原始轮次列，用于后续反向写回），
        # 仅排除内部元数据字段，保证 CSV 列结构在输出端可还原
        sub_row = {
            k: v for k, v in case_row.items()
            if not k.startswith('_')
            and k != 'is_multi_turn'
        }
        # 覆盖为当前轮次的具体值
        sub_row['query'] = question
        sub_row['expected_behavior'] = expected_output
        sub_row['_source_csv'] = source_csv
        sub_row['_turn'] = turn_num
        sub_row['_parent_case_id'] = parent_case_id
        sub_row['_is_multi_turn_sub'] = True
        sub_row['_total_turns'] = len(turns)
        if answer_summary is not None:
            sub_row['agent_response'] = answer_summary
            sub_row['res_time(s)'] = res_time
        sub_rows.append(sub_row)

    ok = sum(1 for r in sub_rows if 'agent_response' in r)
    logger.info(f'[多轮] 完成, 用例编号={parent_case_id}, 成功{ok}/{len(sub_rows)}轮')
    return sub_rows


# 按轮次展开的核心字段
_PER_TURN_CORE = {'query', 'agent_response', 'expected_behavior', 'res_time(s)'}
# 评测结果字段（每轮独立），通过后缀模式匹配
_PER_TURN_EVAL_SUFFIXES = ('_score', '_is_success', '_reason', '_threshold')
# 子行 is_success 保留各轮结论；合并行 is_success 使用 _parent_all_pass。
# 不额外展开 is_success(tN)；每轮明细由 *_is_success(tN) 体现。
_PER_TURN_EVAL_EXACT = {'evaluate_error', 'retry_count'}


def _is_per_turn_field(key: str) -> bool:
    """判断字段是否应按轮次展开（带 (tN) 后缀）"""
    if key in _PER_TURN_CORE:
        return True
    if key in _PER_TURN_EVAL_EXACT:
        return True
    if any(key.endswith(s) for s in _PER_TURN_EVAL_SUFFIXES):
        return True
    return False


def _normalize_single_turn_fields(row: dict) -> dict:
    """为单轮用例的按轮次字段统一添加 (t1) 后缀，与多轮合并行表头一致"""
    normalized = dict(row)
    for key in list(normalized.keys()):
        if _is_per_turn_field(key) and '(t' not in key:
            normalized[f'{key}(t1)'] = normalized.pop(key)
    return normalized


def _merge_multi_turn_rows(rows: list[dict],
                           normalize_single_turn: bool = False) -> list[dict]:
    """
    将同一父用例的多轮子行合并为一行。

    单轮行（无 _parent_case_id）：
      - normalize_single_turn=False: 原样保留（tmp 中间文件，需保持 query 等字段可被 resume 读取）
      - normalize_single_turn=True:  统一添加 (t1) 后缀（最终输出，与多轮合并行表头一致）

    多轮子行按 _parent_case_id 分组，每组合并为一行：
      - 核心字段按轮次展开: query(tN), agent_response(tN), expected_behavior(tN), res_time(s)(tN)
      - 评测字段按轮次展开: *_score(tN), *_is_success(tN), *_reason(tN), *_threshold(tN),
                           evaluate_error(tN), retry_count(tN)
      - 子行 is_success 保留每轮结果；评价后的合并行以 _parent_all_pass 写入整体 is_success
      - 公共字段（CSV 元数据如 用例编号、类别 等）取首个子行的值，不加后缀
      - 保留 _parent_case_id, _total_turns, _is_multi_turn_sub 元数据

    例如: MT001 的 2 个子行合并为:
      {'用例编号':'MT001', '类别':'咨询',
       'query(t1)':'...', 'agent_response(t1)':'...',
       'reverse_validation_score(t1)':0.8, 'reverse_validation_is_success(t1)':True, ...,
       'query(t2)':'...', 'agent_response(t2)':'...',
       'reverse_validation_score(t2)':0.3, 'reverse_validation_is_success(t2)':False, ...,
       'is_success': False, '用例是否通过': False,
       '_parent_case_id':'MT001', '_total_turns':2, '_is_multi_turn_sub':True}
    """
    single_rows = [r for r in rows if not r.get('_parent_case_id')]
    multi_rows = [r for r in rows if r.get('_parent_case_id')]

    if not multi_rows:
        return rows

    # 按 _parent_case_id 分组
    groups: dict[str, list[dict]] = {}
    for r in multi_rows:
        pid = r['_parent_case_id']
        groups.setdefault(pid, []).append(r)

    merged_rows = []
    for pid, sub_rows in groups.items():
        sub_rows.sort(key=lambda r: r.get('_turn', 0))

        merged = {}
        # 公共字段：首个子行中非 _ 开头、非按轮次展开的字段（CSV 元数据如 用例编号、类别等）
        for key, val in sub_rows[0].items():
            if (not key.startswith('_')
                    and key not in _PER_TURN_CORE
                    and key not in _PER_TURN_EVAL_EXACT
                    and not any(key.endswith(s) for s in _PER_TURN_EVAL_SUFFIXES)):
                merged[key] = val

        merged['_parent_case_id'] = pid
        merged['_total_turns'] = sub_rows[0].get('_total_turns', len(sub_rows))
        merged['_is_multi_turn_sub'] = True
        merged['_source_csv'] = sub_rows[0].get('_source_csv', '')
        if '_parent_all_pass' in sub_rows[0]:
            merged['_parent_all_pass'] = sub_rows[0]['_parent_all_pass']
            # 整体结论取聚合结果，不能沿用第一轮的通过状态。
            # Agent 阶段写 tmp 时尚未聚合，不额外生成评价结果。
            merged['is_success'] = merged['_parent_all_pass']

        # 每轮字段展开（核心 + 评测结果）
        for i, sub in enumerate(sub_rows, start=1):
            t = f'(t{i})'
            for key, val in sub.items():
                if _is_per_turn_field(key):
                    merged[f'{key}{t}'] = val

        # 反向归一化：从各指标 *_is_success(tN) 取 AND 写回第N轮对话是否通过
        for i in range(1, len(sub_rows) + 1):
            turn_metric_keys = [
                k for k in merged
                if k.endswith(f'_is_success(t{i})') and '_model' not in k
            ]
            if turn_metric_keys:
                merged[f'第{i}轮对话是否通过'] = all(
                    str(merged[k]).strip().upper() == 'TRUE' for k in turn_metric_keys
                )

        # 反向归一化：is_success(整体) → 用例是否通过
        if 'is_success' in merged:
            merged['用例是否通过'] = merged['is_success']

        merged_rows.append(merged)

    if normalize_single_turn:
        single_rows = [_normalize_single_turn_fields(r) for r in single_rows]
    return single_rows + merged_rows


def _group_evaluation_rows(rows: list[dict]) -> list[list[dict]]:
    """按最终输出顺序分组；同一多轮父用例不能跨评价/落盘批次。"""
    singles = []
    parents = {}
    for row in rows:
        parent_id = row.get('_parent_case_id')
        if parent_id:
            parents.setdefault(parent_id, []).append(row)
        else:
            singles.append([row])
    return singles + list(parents.values())


def _aggregate_multi_turn(test_case: dict):
    """
    多轮对话聚合判定：全部轮次 is_success 为 True 则父用例整体通过；
    仅写入 _parent_all_pass，不改写每轮的 is_success 或 evaluate_error。
    合并输出时，再将父用例结论写入最终行的 is_success。
    """
    csv_rows = test_case['csv']

    # 按父用例分组
    multi_groups: dict[str, list[dict]] = {}
    for row in csv_rows:
        pid = row.get('_parent_case_id')
        if pid:
            multi_groups.setdefault(pid, []).append(row)

    if not multi_groups:
        return

    for parent_id, sub_rows in multi_groups.items():
        all_pass = all(
            row.get('is_success') in (True, 'True') for row in sub_rows
        )

        for row in sub_rows:
            row['_parent_all_pass'] = all_pass

        logger.info(
            f'[聚合] 父用例={parent_id}: '
            f'{"全部通过" if all_pass else "存在未通过轮次"}, 共{len(sub_rows)}轮'
        )


#评测流水线
class EvaluationPipeline:
    """评测流水线，按阶段串联：准备 → Agent调用 → 评测 → 报告"""

    def __init__(self, csv_path: str | None, metrics: list[str] | None, resume: bool = False,
                 agent_classes: list[str] | None = None, seed: int | None = None,
                 fill_preview: bool = False, refresh_entities: bool = False,
                 resume_result_dir: str | None = None):
        self.conf = ConfigReader.get_instance()
        self.csv_path = csv_path
        self.metrics = metrics
        self.resume = resume
        self.agent_classes = agent_classes
        self.seed = seed
        self.fill_preview = fill_preview
        self.refresh_entities = refresh_entities
        self.resume_result_dir = resume_result_dir
        if resume_result_dir is not None and not resume:
            raise ValueError('--resume-result-dir 必须与 --resume 一起使用')

    def _should_refresh_entities(self) -> bool:
        """是否刷新实体映射表：CLI --refresh-entities 与配置项取 OR"""
        conf_enabled = self.conf.get(
            'dataset.placeholder_fill.refresh_entities', False)
        return bool(self.refresh_entities or conf_enabled)

    def _is_agent_only(self, agent_class: str) -> bool:
        """是否仅调用指定 Agent 并在 tmp 中间文件写入后结束。"""
        return is_agent_only_enabled(agent_class, self.conf)

    def _refresh_entities(self):
        """刷新实体映射表：调用业务平台接口更新 entity_mapping.json。

        登录或任一采集接口失败时抛 RuntimeError 中断执行（fail fast），
        避免静默使用过期实体数据导致评测结果失真。
        """
        mapping_path = self.conf.get('dataset.placeholder_fill.entity_mapping_path', None)
        if not mapping_path:
            raise ValueError(
                '--refresh-entities 需要配置 dataset.placeholder_fill.entity_mapping_path'
            )
        logger.info(f'[实体刷新] 开始调用业务平台接口更新实体映射: {mapping_path}')
        from tool.business_platform_client import BusinessPlatformClient
        BusinessPlatformClient().export_entity_mapping(mapping_path)
        logger.info(f'[实体刷新] 实体映射更新完成: {mapping_path}')

    def run(self):
        if self.fill_preview:
            if self._should_refresh_entities():
                self._refresh_entities()
            self._run_fill_preview()
            return
        if self.resume:
            test_cases_by_class = self._prepare_from_tmp()
            # resume 不重新填充，seed 从 tmp meta.json 读取用于报告输出
            self.seed = self._extract_seed_from_cases(test_cases_by_class)
            result_dir = getattr(self, 'resume_result_dir', None)
            test_cases_by_class, summary = prepare_resume(test_cases_by_class, result_dir)
            mode = '增量恢复' if result_dir is not None else '全量重新评价'
            message = (f'[resume] {mode}：共 {summary.total} 条，复用 {summary.reused} 条，'
                       f'待评价 {summary.reevaluate} 条，不可评价 {summary.unavailable} 条')
            logger.info(message)
            print(message, flush=True)
            if result_dir is not None and summary.all_passed:
                message = '[resume] 全部通过，无需恢复评价；未生成新结果和报告'
                logger.info(message)
                print(message, flush=True)
                return
            self._invoke_agents_resume(test_cases_by_class)
        else:
            if self._should_refresh_entities():
                self._refresh_entities()
            if self.csv_path is None:
                # 嗅探模式：先扫描 raw 待替换模板 → 填充 → 生成 output_dir 可执行数据集
                self.seed = fill_raw_datasets(self.seed)
            test_cases_by_class = self._prepare()
            # 占位符填充（仅命中 target_dirs 的数据集；output_dir 产物自动跳过），
            # 返回实际使用的 seed
            self.seed = fill_test_cases(test_cases_by_class, self.seed)
            test_cases_by_class = self._invoke_agents(test_cases_by_class)
            if not test_cases_by_class:
                logger.info('[agent-only] Agent 调用结果已写入 tmp，流水线结束')
                print('[agent-only] 完成：Agent 调用结果已写入 tmp，未执行评测与报告生成')
                return
        base_path = self._evaluate(test_cases_by_class)
        self._report(base_path, self.seed)

    # --fill-preview: 仅填充并输出 *.filled.csv，不执行评测
    def _run_fill_preview(self):
        seed = preview_fill(self.csv_path, self.seed)
        print(f'[填充预览] 完成，seed={seed}，可使用同 seed 重跑评测复现测试数据')
        logger.info(f'[填充预览] 完成，seed={seed}')

    @staticmethod
    def _extract_seed_from_cases(test_cases_by_class: dict) -> int | None:
        """从 tmp 加载的 test_case 中提取 meta.json 保存的 seed"""
        for tc_list in test_cases_by_class.values():
            for tc in tc_list:
                seed = tc.get('seed')
                if seed is not None:
                    return seed
        return None

    #阶段1: 用例准备
    def _prepare(self) -> dict[str, list[dict]]:
        return make_test_case_list(self.csv_path, self.metrics, self.agent_classes)

    #阶段2: Agent 并发调用 + 组装 LLM 用例
    def _invoke_agents(
        self,
        test_cases_by_class: dict[str, list[dict]],
    ) -> dict[str, list[dict]]:
        """
        按 agent 类隔离执行：每个类只处理自己子目录下的用例。

        '__shared__' key 表示 CLI -cp 指定的共享用例，所有 enabled agent 均执行。

        :param test_cases_by_class: {class_name: [{csv, case_name, metrics, agent_class}, ...], ...}
        """
        enabled_classes = get_enabled_classes(self.agent_classes)
        logger.info(f"[阶段2] 启用的 agent 类: {enabled_classes}")
        evaluation_cases_by_class: dict[str, list[dict]] = {}

        max_worker = self.conf.get("agents.http_agent.call_agent_th_max")
        submit_delay = self.conf.get("agents.http_agent.submit_delay", 0)

        for class_name, class_test_cases in test_cases_by_class.items():
            # 确定该组用例由哪些 agent 执行
            if class_name == '__shared__':
                target_agents = enabled_classes
            else:
                target_agents = [class_name]

            agent_only_modes = {
                agent_cls: self._is_agent_only(agent_cls)
                for agent_cls in target_agents
            }
            if len(set(agent_only_modes.values())) > 1:
                raise ValueError(
                    '共享测试集不能同时用于 agent-only 与正常评测 Agent；'
                    '请通过 -a 分开执行。当前配置: '
                    f'{agent_only_modes}'
                )
            is_agent_only = bool(target_agents) and all(agent_only_modes.values())

            # 展平本组用例，构建 case_name → metrics 映射
            # 同时预计算 _tenant_id、_is_direct_inquiry 等标，避免下游逐 case 反查
            metrics_by_case = {}
            all_cases = []
            for tc in class_test_cases:
                source = tc['case_name']
                metrics = tc['metrics']
                tenant_id = _resolve_tenant_id(source)
                is_direct = 'direct_inquiry' in source.lower()
                need_retrieval = bool(RETRIEVAL_METRIC_NAMES & set(metrics))
                metrics_by_case[source] = metrics
                for case in tc['csv']:
                    case['_source_csv'] = source
                    case['_tenant_id'] = tenant_id
                    case['_is_direct_inquiry'] = is_direct
                    case['_need_retrieval'] = need_retrieval
                    all_cases.append(case)

            # 字段归一化：推理数据集 第1轮→query, 第1轮对话预期结果→expected_behavior
            for case in all_cases:
                if 'query' not in case and '第1轮' in case:
                    case['query'] = case['第1轮']
                    case['expected_behavior'] = case.get('第1轮对话预期结果', '')

            # 保存原始用例，多 agent 共享时每个 agent 基于独立副本执行
            _original_all_cases = all_cases

            # 每个目标 agent 执行本组用例
            for agent_cls in target_agents:
                agent = create_agent(agent_cls)

                # 分离多轮 / 单轮（每个 agent 基于独立副本，避免交叉覆盖）
                multi_cases = [dict(c) for c in _original_all_cases if _is_multi_turn(c)]
                single_cases = [dict(c) for c in _original_all_cases if not _is_multi_turn(c)]

                logger.info(
                    f'[阶段2] {agent_cls} ← 用例组 "{class_name}" '
                    f'(单轮 {len(single_cases)} + 多轮 {len(multi_cases)} = '
                    f'{len(_original_all_cases)} 条)'
                )

                expanded_multi: list[dict] = []

                # 单轮：DataQA direct_inquiry 走两段式，其他 agent 走标准调用
                if single_cases:
                    if agent_cls == 'DataQA':
                        direct_cases = [
                            c for c in single_cases
                            if c.get('_is_direct_inquiry')
                        ]
                        other_cases = [
                            c for c in single_cases if c not in direct_cases
                        ]
                        if other_cases:
                            raise NotImplementedError(
                                f'DataQA 非 direct_inquiry 用例暂不支持: '
                                f'{[(c.get("用例编号", c.get("query", "")[:40])) for c in other_cases]}'
                            )
                        if direct_cases:
                            run_in_thread_pool(
                                partial(_call_agent_dataqa_direct, agent),
                                direct_cases,
                                max_workers=max_worker,
                                submit_delay=submit_delay,
                                task_name=f"DataQA-direct({agent_cls}←{class_name})"
                            )
                    else:
                        run_in_thread_pool(
                            partial(call_agent, agent), single_cases,
                            max_workers=max_worker, submit_delay=submit_delay,
                            task_name=f"call_agent({agent_cls}←{class_name})"
                        )

                    # 检索轨：metrics 中包含检索类指标时，额外调 RAGFlow
                    retrieval_cases = [
                        c for c in single_cases if c.get('_need_retrieval')
                    ]
                    if retrieval_cases:
                        retriever = create_agent('RAGFlowRetriever')
                        run_in_thread_pool(
                            partial(call_ragflow_retrieve, retriever),
                            retrieval_cases,
                            max_workers=max_worker, submit_delay=submit_delay,
                            task_name=f"RAGFlow检索({agent_cls}←{class_name})"
                        )

                # 多轮：每条用例内部串行（同 session_id），用例间并行
                if multi_cases:
                    with ThreadPoolExecutor(max_workers=max_worker) as executor:
                        multi_futures = {}
                        for case in multi_cases:
                            multi_futures[
                                executor.submit(call_multi_turn_agent, agent, case)
                            ] = case

                        for future in as_completed(multi_futures):
                            try:
                                sub_rows = future.result()
                                expanded_multi.extend(sub_rows)
                            except Exception as e:
                                original = multi_futures[future]
                                logger.error(
                                    f'[多轮] 用例处理异常, '
                                    f'用例编号={original.get("用例编号", "?")}, 错误: {e}'
                                )

                # 合并：单轮（原地已修改）+ 多轮展开子行
                all_cases = single_cases + expanded_multi

            # 将展开后的结果按 _source_csv 单次遍历分组，回写 test_cases_by_class
            grouped_by_source: dict[str, list[dict]] = {}
            for case in all_cases:
                src = case['_source_csv']
                if src not in grouped_by_source:
                    grouped_by_source[src] = []
                grouped_by_source[src].append(case)

            for tc in class_test_cases:
                tc['csv'] = grouped_by_source.get(tc['case_name'], [])

            # 多轮子行合并为一行（仅用于 tmp 写入，不影响后续 make_llm_case）
            all_cases_for_tmp = _merge_multi_turn_rows(all_cases)

            # 按原始 CSV 分组写入 tmp 中间文件
            groups: dict[str, list[dict]] = {}
            for case in all_cases_for_tmp:
                src = case['_source_csv']
                if src not in groups:
                    groups[src] = []
                groups[src].append(case)

            for src_path, rows in groups.items():
                src = Path(src_path)
                tmp_dir = src.parent / 'tmp'
                tmp_dir.mkdir(parents=True, exist_ok=True)
                stem = src.stem

                CsvWriter(tmp_dir / f'{stem}_tmp.csv').write_from_test_cases(rows)

                meta = {
                    'case_name': src_path,
                    'metrics': metrics_by_case.get(src_path, []),
                    'seed': self.seed,
                }
                meta_path = tmp_dir / f'{stem}_tmp.meta.json'
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

                logger.info(f'[tmp] 已保存中间文件: {meta_path}')

            # agent-only 在 Agent 响应已落盘后截断，不组装 LLMTestCase，也不进入评测。
            if is_agent_only:
                logger.info(
                    f'[agent-only] {class_name}: Agent 调用完成且结果已写入 tmp，'
                    '跳过 LLMTestCase 组装与评测'
                )
                continue

            # 组装 LLM 用例 — 按用例类型分流（skip_assertion 用例不参与组装与断言）
            dataqa_cases = [c for c in all_cases
                            if 'actual_capability_id' in c or 'actual_parameters' in c]
            other_cases = [c for c in all_cases
                           if c not in dataqa_cases and not _is_skip_assertion(c)]

            if other_cases:
                run_in_thread_pool(
                    make_llm_case, other_cases,
                    max_workers=max_worker, task_name=f'make_llm_test_case({class_name})'
                )

            for case in dataqa_cases:
                make_llm_case_structured(case)

            # 检索轨 LLM 用例组装
            retrieval_cases = [
                c for c in all_cases if c.get('_need_retrieval')
            ]
            if retrieval_cases:
                run_in_thread_pool(
                    make_llm_case_retrieval, retrieval_cases,
                    max_workers=max_worker,
                    task_name=f'make_llm_retrieval({class_name})'
                )

            ok_dataqa = sum(1 for c in dataqa_cases
                            if 'llm_test_case_capability' in c and 'llm_test_case_params' in c)
            ng_dataqa = len(dataqa_cases) - ok_dataqa
            ok_other = sum(1 for c in other_cases if 'llm_test_case' in c)
            ng_other = len(other_cases) - ok_other
            ok_retrieval = sum(1 for c in retrieval_cases
                               if 'llm_test_case_retrieval' in c)
            ng_retrieval = len(retrieval_cases) - ok_retrieval

            total_ok = ok_dataqa + ok_other + ok_retrieval
            total_ng = ng_dataqa + ng_other + ng_retrieval
            skip_cnt = sum(1 for c in all_cases if _is_skip_assertion(c))
            if total_ng or skip_cnt:
                parts = []
                if ng_other:
                    parts.append(f"文本质量 {ok_other}/{len(other_cases)}")
                if ng_dataqa:
                    parts.append(f"结构化 {ok_dataqa}/{len(dataqa_cases)}")
                if ng_retrieval:
                    parts.append(f"检索 {ok_retrieval}/{len(retrieval_cases)}")
                if skip_cnt:
                    parts.append(f"跳过断言 {skip_cnt}")
                logger.info(
                    f"[阶段2] {class_name}: Agent调用/用例组装 "
                    f"成功 {total_ok}, 失败 {total_ng}, 跳过断言 {skip_cnt} "
                    f"(共 {len(all_cases)}) — {', '.join(parts)}"
                )
            else:
                logger.info(
                    f"[阶段2] {class_name}: Agent调用/用例组装 全部成功 "
                    f"({len(all_cases)} 条)"
                )

            evaluation_cases_by_class[class_name] = class_test_cases

        return evaluation_cases_by_class

    # resume: 从 tmp 目录加载中间文件
    def _prepare_from_tmp(self) -> dict[str, list[dict]]:
        return make_tmp_test_case_list(self.csv_path, self.metrics, self.agent_classes)

    # resume: 跳过 call_agent，仅执行 llm_test_case 组装
    def _invoke_agents_resume(self, test_cases_by_class: dict[str, list[dict]]):
        max_worker = self.conf.get("agents.http_agent.call_agent_th_max")

        for class_name, class_test_cases in test_cases_by_class.items():
            all_cases = [
                case for tc in class_test_cases for case in tc['csv']
            ]

            if not all_cases:
                logger.warning(f'[resume] {class_name}: 未找到可恢复的测试用例')
                continue

            # 组装 LLM 用例 — 按用例类型分流（skip_assertion 用例不参与组装与断言）
            dataqa_cases = [c for c in all_cases
                            if 'actual_capability_id' in c or 'actual_parameters' in c]
            other_cases = [c for c in all_cases
                           if c not in dataqa_cases and not _is_skip_assertion(c)]

            if other_cases:
                run_in_thread_pool(
                    make_llm_case, other_cases,
                    max_workers=max_worker,
                    task_name=f'make_llm_test_case({class_name})'
                )

            for case in dataqa_cases:
                make_llm_case_structured(case)

            # 检索轨 LLM 用例组装
            retrieval_cases = [
                c for c in all_cases if c.get('_need_retrieval')
            ]
            if retrieval_cases:
                run_in_thread_pool(
                    make_llm_case_retrieval, retrieval_cases,
                    max_workers=max_worker,
                    task_name=f'make_llm_retrieval_resume({class_name})'
                )

            ok_dataqa = sum(1 for c in dataqa_cases
                            if 'llm_test_case_capability' in c and 'llm_test_case_params' in c)
            ng_dataqa = len(dataqa_cases) - ok_dataqa
            ok_other = sum(1 for c in other_cases if 'llm_test_case' in c)
            ng_other = len(other_cases) - ok_other
            ok_retrieval = sum(1 for c in retrieval_cases
                               if 'llm_test_case_retrieval' in c)
            ng_retrieval = len(retrieval_cases) - ok_retrieval

            total_ok = ok_dataqa + ok_other + ok_retrieval
            total_ng = ng_dataqa + ng_other + ng_retrieval
            skip_cnt = sum(1 for c in all_cases if _is_skip_assertion(c))
            if total_ng or skip_cnt:
                parts = []
                if ng_other:
                    parts.append(f"文本质量 {ok_other}/{len(other_cases)}")
                if ng_dataqa:
                    parts.append(f"结构化 {ok_dataqa}/{len(dataqa_cases)}")
                if ng_retrieval:
                    parts.append(f"检索 {ok_retrieval}/{len(retrieval_cases)}")
                if skip_cnt:
                    parts.append(f"跳过断言 {skip_cnt}")
                logger.info(
                    f"[resume] {class_name}: llm_test_case组装 "
                    f"成功 {total_ok}, 失败 {total_ng}, 跳过断言 {skip_cnt} "
                    f"(共 {len(all_cases)}) — {', '.join(parts)}"
                )
            else:
                logger.info(
                    f"[resume] {class_name}: llm_test_case组装 全部成功 "
                    f"({len(all_cases)} 条)"
                )

    #阶段3: 评测执行 + 结果写入（按 agent 类分目录输出）
    def _evaluate(self, test_cases_by_class: dict[str, list[dict]]) -> str:
        batch_size = self.conf.get('result.write_batch_size', 20)
        if type(batch_size) is not int or not 10 <= batch_size <= 30:
            raise ValueError('result.write_batch_size 必须是 10–30 之间的整数')
        save_path = self.conf.get("result.save_path")
        base_path = mkdir_with_timestamp(save_path)

        for class_name, class_cases in test_cases_by_class.items():
            dir_name = 'default' if class_name == '__shared__' else class_name.lower()
            class_dir = os.path.join(base_path, dir_name)
            os.makedirs(class_dir, exist_ok=True)

            logger.info(f'[阶段3] {class_name} → {class_dir} ({len(class_cases)} 组用例)')

            writer = AsyncResultWriter(class_dir)
            writer.start()

            try:
                for test_case in class_cases:
                    groups = _group_evaluation_rows(test_case['csv'])
                    has_multi_turn = (test_case.get('resume_has_multi_turn', False)
                                      or any(r.get('_parent_case_id') for r in test_case['csv']))
                    completed_rows = []
                    baseline = test_case.get('reused_rows', []) + test_case.get('skipped_rows', [])
                    baseline = [dict(row) for row in baseline]
                    if has_multi_turn:
                        baseline = [_normalize_single_turn_fields(row) if not row.get('_parent_case_id')
                                    else row for row in baseline]
                    total = len(groups) + len(baseline)
                    # 复用记录和不可评价记录不进入任何评价函数，但同样完整提交。
                    for start in range(0, len(baseline), batch_size):
                        saved = baseline[start:start + batch_size]
                        writer.submit({'csv': saved}, test_case['case_name'], append=bool(completed_rows))
                        writer.flush()
                        completed_rows.extend(saved)
                    for offset in range(0, len(groups), batch_size):
                        rows = [row for group in groups[offset:offset + batch_size]
                                for row in group]
                        batch = {**test_case, 'csv': rows}
                        logger.info(
                            f"[阶段3] {test_case['case_name']} "
                            f"评价第 {offset + 1}–{min(offset + batch_size, len(groups))}/{len(groups)} 条"
                        )
                        # 无有效评价结果时不得残留输入中的旧整体通过标志。
                        for row in rows:
                            row['is_success'] = False
                        run_evaluate(batch)
                        run_evaluate_structured(batch)
                        # 初评只写指标结果，复核筛选前须先汇总本轮通过状态。
                        recompute_overall_success(batch)
                        run_multimodel_reevaluate(batch)
                        # 先汇总复核后的每轮状态，再独立计算父用例整体状态。
                        recompute_overall_success(batch)
                        for row in rows:
                            row['用例是否通过'] = row.get('is_success') is True
                        _aggregate_multi_turn(batch)
                        output_rows = _merge_multi_turn_rows(batch['csv'],
                                                             normalize_single_turn=True)
                        # 混合数据集中，即使本批全是单轮，也须沿用整表 (t1) 列格式。
                        if has_multi_turn and not any(r.get('_parent_case_id') for r in batch['csv']):
                            output_rows = [_normalize_single_turn_fields(r) for r in output_rows]
                        batch['csv'] = output_rows
                        writer.submit(batch, test_case['case_name'], append=bool(completed_rows))
                        # 明确的落盘边界：写入失败时停止，不继续消耗模型请求。
                        writer.flush()
                        completed_rows.extend(output_rows)
                        logger.info(
                            f"[阶段3] {test_case['case_name']} 已落盘 {len(completed_rows)}/{total} 条"
                        )
                    test_case['csv'] = completed_rows
            finally:
                # 评价异常/用户中断时，也收尾已提交批次。
                writer.wait_and_stop()
            print(f'[{class_name}] {writer.get_stats()}')

        return base_path

    #阶段4: 报告生成（遍历各 agent 子目录）
    def _report(self, base_path: str, seed: int | None = None):
        for entry in sorted(os.listdir(base_path)):
            class_dir = os.path.join(base_path, entry)
            if not os.path.isdir(class_dir):
                continue

            # 提取失败用例
            extract_bad_cases(class_dir)

            # 生成评测报告
            report_path = os.path.join(class_dir, 'test_report.md')
            md_writer = MarkdownWriter(report_path, seed=seed)

            for file in sorted(os.listdir(class_dir)):
                if file.startswith('result_outputs_') and file.endswith('.csv'):
                    cr = CollectionResult(os.path.join(class_dir, file))
                    md_writer.write_report(
                        file, cr.task_success_stats(), cr.res_time()
                    )

            logger.info(f'[阶段4] 报告已生成: {report_path}')


#入口
if __name__ == "__main__":
    params = parse_args()
    EvaluationPipeline(
        params.csv_path, params.metrics,
        resume=params.resume, agent_classes=params.agent_classes,
        resume_result_dir=params.resume_result_dir,
        seed=params.seed, fill_preview=params.fill_preview,
        refresh_entities=params.refresh_entities
    ).run()
