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
from tool.concurrency import run_in_thread_pool
from tool.csv_writer import CsvWriter
from tool.file_utils import mkdir_with_timestamp
from tool import AsyncResultWriter, MarkdownWriter
from tool.collection_result import CollectionResult
from tool.get_bad_cases import extract_bad_cases
from tool.log_factory import LogFactory
from agents.factory import create_agent, get_enabled_classes
from pipeline.test_case_loader import make_test_case_list, make_tmp_test_case_list
from evaluator.runner import run_evaluate, run_evaluate_structured, run_multimodel_reevaluate, recompute_overall_success

logger = LogFactory.get_logger(__name__)

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
        '-a', '--agent_classes',
        nargs='+',
        help='指定调用的 agent 类名，支持多个，如: -a Diagnosis 或 -a PVAssistant Diagnosis'
    )
    return parser.parse_args()


#Agent 调用薄函数
def call_agent(agent, test_case_csv: dict):
    """agent调用入口，内置重试逻辑，成功时将响应并入字典"""
    conf = ConfigReader.get_instance()
    max_retries = conf.get("agents.http_agent.call_agent_max_retries", 3)
    retry_delay = conf.get("agents.http_agent.call_agent_retry_delay", 2)

    query = test_case_csv.get('query', '')
    logger.info(f'call_agent, query: {query[:80]}')

    for attempt in range(max_retries + 1):
        try:
            answer_raw, answer_summary, res_time = agent.call_agent(query)
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
                time.sleep(retry_delay)
            else:
                logger.error(
                    f'call_agent最终失败（已重试{max_retries}次）, '
                    f'query: {query[:80]}, 错误: {e}'
                )


def _call_agent_dataqa_direct(agent, test_case_csv: dict):
    """DataQA direct_inquiry 两段式调用: new_query → confirm_execute，内置重试"""
    conf = ConfigReader.get_instance()
    max_retries = conf.get("agents.http_agent.call_agent_max_retries", 3)
    retry_delay = conf.get("agents.http_agent.call_agent_retry_delay", 2)

    query = test_case_csv.get('query', '')
    logger.info(f'DataQA direct_inquiry, query: {query[:80]}')

    for attempt in range(max_retries + 1):
        try:
            start = time.perf_counter()
            new_query_raw, confirmation_id, session_id = agent.new_query(query)
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
                time.sleep(retry_delay)
            else:
                logger.error(
                    f'DataQA direct_inquiry 最终失败（已重试{max_retries}次）, '
                    f'query: {query[:80]}, 错误: {e}'
                )


def make_llm_case(test_case_csv: dict):
    """llm_case组装方法，组装完成的用例重新放回字典中

    agent_response 缺失时（call_agent 失败）跳过该用例，写入错误标记。
    """
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


# ---- 多轮对话支持 ----

def _is_multi_turn(case_row: dict) -> bool:
    """判断用例是否为多轮对话，兼容CSV字符串和Python bool"""
    val = case_row.get('is_multi_turn', False)
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
    logger.info(f'[多轮] 开始, 用例编号={parent_case_id}, 共{len(turns)}轮, session_id={session_id}')

    sub_rows = []
    for turn_num, question, expected_output in turns:
        logger.info(f'[多轮] 用例={parent_case_id}, 第{turn_num}轮, query: {question[:80]}')

        answer_summary = None
        res_time = None
        for attempt in range(max_retries + 1):
            try:
                answer_raw, answer_summary, res_time = agent.call_agent(
                    question, session_id=session_id
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
# 注意: is_success 是整体标识（_aggregate_multi_turn 已将各子行统一），
# 不应按轮次展开为 is_success(tN)；每轮明细由 *_is_success(tN) 体现。
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
      - is_success 为整体标识，不按轮次展开
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


def _aggregate_multi_turn(test_case: dict):
    """
    多轮对话聚合判定：全部轮次 is_success 为 True 则父用例整体通过；
    任一未通过则所有子行标记 is_success=False，提取 bad_case 时自然落入。
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
            if not all_pass:
                row['is_success'] = False
                existing_error = row.get('evaluate_error', '') or ''
                if existing_error:
                    row['evaluate_error'] = (
                        f'{existing_error}; '
                        f'父用例[{parent_id}]存在未通过轮次，整体判定不通过'
                    )
                else:
                    row['evaluate_error'] = (
                        f'父用例[{parent_id}]存在未通过轮次，整体判定不通过'
                    )

        logger.info(
            f'[聚合] 父用例={parent_id}: '
            f'{"全部通过" if all_pass else "存在未通过轮次"}, 共{len(sub_rows)}轮'
        )


#评测流水线
class EvaluationPipeline:
    """评测流水线，按阶段串联：准备 → Agent调用 → 评测 → 报告"""

    def __init__(self, csv_path: str | None, metrics: list[str] | None, resume: bool = False,
                 agent_classes: list[str] | None = None):
        self.conf = ConfigReader.get_instance()
        self.csv_path = csv_path
        self.metrics = metrics
        self.resume = resume
        self.agent_classes = agent_classes

    def run(self):
        if self.resume:
            test_cases_by_class = self._prepare_from_tmp()
            self._invoke_agents_resume(test_cases_by_class)
        else:
            test_cases_by_class = self._prepare()
            self._invoke_agents(test_cases_by_class)
        base_path = self._evaluate(test_cases_by_class)
        self._report(base_path)

    #阶段1: 用例准备
    def _prepare(self) -> dict[str, list[dict]]:
        return make_test_case_list(self.csv_path, self.metrics, self.agent_classes)

    #阶段2: Agent 并发调用 + 组装 LLM 用例
    def _invoke_agents(self, test_cases_by_class: dict[str, list[dict]]):
        """
        按 agent 类隔离执行：每个类只处理自己子目录下的用例。

        '__shared__' key 表示 CLI -cp 指定的共享用例，所有 enabled agent 均执行。

        :param test_cases_by_class: {class_name: [{csv, case_name, metrics, agent_class}, ...], ...}
        """
        enabled_classes = get_enabled_classes(self.agent_classes)
        logger.info(f"[阶段2] 启用的 agent 类: {enabled_classes}")

        max_worker = self.conf.get("agents.http_agent.call_agent_th_max")
        submit_delay = self.conf.get("agents.http_agent.submit_delay", 0)

        for class_name, class_test_cases in test_cases_by_class.items():
            # 确定该组用例由哪些 agent 执行
            if class_name == '__shared__':
                target_agents = enabled_classes
            else:
                target_agents = [class_name]

            # 展平本组用例，构建 case_name → metrics 映射
            metrics_by_case = {}
            all_cases = []
            for tc in class_test_cases:
                metrics_by_case[tc['case_name']] = tc['metrics']
                for case in tc['csv']:
                    case['_source_csv'] = tc['case_name']
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
                            if 'direct_inquiry' in c.get('_source_csv', '').lower()
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

            # 将展开后的结果同步回 test_cases_by_class 结构
            for tc in class_test_cases:
                source = tc['case_name']
                tc['csv'] = [c for c in all_cases if c.get('_source_csv') == source]

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
                    'metrics': metrics_by_case.get(src_path, [])
                }
                meta_path = tmp_dir / f'{stem}_tmp.meta.json'
                meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')

                logger.info(f'[tmp] 已保存中间文件: {meta_path}')

            # 组装 LLM 用例 — 按用例类型分流
            dataqa_cases = [c for c in all_cases
                            if 'actual_capability_id' in c or 'actual_parameters' in c]
            other_cases = [c for c in all_cases if c not in dataqa_cases]

            if other_cases:
                run_in_thread_pool(
                    make_llm_case, other_cases,
                    max_workers=max_worker, task_name=f'make_llm_test_case({class_name})'
                )

            for case in dataqa_cases:
                make_llm_case_structured(case)

            ok_dataqa = sum(1 for c in dataqa_cases
                            if 'llm_test_case_capability' in c and 'llm_test_case_params' in c)
            ng_dataqa = len(dataqa_cases) - ok_dataqa
            ok_other = sum(1 for c in other_cases if 'llm_test_case' in c)
            ng_other = len(other_cases) - ok_other

            total_ok = ok_dataqa + ok_other
            total_ng = ng_dataqa + ng_other
            if total_ng:
                parts = []
                if ng_other:
                    parts.append(f"文本质量 {ok_other}/{len(other_cases)}")
                if ng_dataqa:
                    parts.append(f"结构化 {ok_dataqa}/{len(dataqa_cases)}")
                logger.info(
                    f"[阶段2] {class_name}: Agent调用/用例组装 "
                    f"成功 {total_ok}, 失败 {total_ng} (共 {len(all_cases)}) — "
                    f"{', '.join(parts)}"
                )
            else:
                logger.info(
                    f"[阶段2] {class_name}: Agent调用/用例组装 全部成功 "
                    f"({len(all_cases)} 条)"
                )

    # resume: 从 tmp 目录加载中间文件
    def _prepare_from_tmp(self) -> dict[str, list[dict]]:
        return make_tmp_test_case_list(self.csv_path, self.metrics)

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

            # 组装 LLM 用例 — 按用例类型分流
            dataqa_cases = [c for c in all_cases
                            if 'actual_capability_id' in c or 'actual_parameters' in c]
            other_cases = [c for c in all_cases if c not in dataqa_cases]

            if other_cases:
                run_in_thread_pool(
                    make_llm_case, other_cases,
                    max_workers=max_worker,
                    task_name=f'make_llm_test_case({class_name})'
                )

            for case in dataqa_cases:
                make_llm_case_structured(case)

            ok_dataqa = sum(1 for c in dataqa_cases
                            if 'llm_test_case_capability' in c and 'llm_test_case_params' in c)
            ng_dataqa = len(dataqa_cases) - ok_dataqa
            ok_other = sum(1 for c in other_cases if 'llm_test_case' in c)
            ng_other = len(other_cases) - ok_other

            total_ok = ok_dataqa + ok_other
            total_ng = ng_dataqa + ng_other
            if total_ng:
                parts = []
                if ng_other:
                    parts.append(f"文本质量 {ok_other}/{len(other_cases)}")
                if ng_dataqa:
                    parts.append(f"结构化 {ok_dataqa}/{len(dataqa_cases)}")
                logger.info(
                    f"[resume] {class_name}: llm_test_case组装 "
                    f"成功 {total_ok}, 失败 {total_ng} (共 {len(all_cases)}) — "
                    f"{', '.join(parts)}"
                )
            else:
                logger.info(
                    f"[resume] {class_name}: llm_test_case组装 全部成功 "
                    f"({len(all_cases)} 条)"
                )

    #阶段3: 评测执行 + 结果写入（按 agent 类分目录输出）
    def _evaluate(self, test_cases_by_class: dict[str, list[dict]]) -> str:
        save_path = self.conf.get("result.save_path")
        base_path = mkdir_with_timestamp(save_path)

        for class_name, class_cases in test_cases_by_class.items():
            dir_name = 'default' if class_name == '__shared__' else class_name.lower()
            class_dir = os.path.join(base_path, dir_name)
            os.makedirs(class_dir, exist_ok=True)

            logger.info(f'[阶段3] {class_name} → {class_dir} ({len(class_cases)} 组用例)')

            writer = AsyncResultWriter(class_dir)
            writer.start()

            for test_case in class_cases:
                run_evaluate(test_case)
                run_evaluate_structured(test_case)
                run_multimodel_reevaluate(test_case)
                _aggregate_multi_turn(test_case)
                recompute_overall_success(test_case)
                test_case['csv'] = _merge_multi_turn_rows(test_case['csv'],
                                                          normalize_single_turn=True)
                writer.submit(test_case, test_case['case_name'])

            writer.wait_and_stop()
            print(f'[{class_name}] {writer.get_stats()}')

        return base_path

    #阶段4: 报告生成（遍历各 agent 子目录）
    def _report(self, base_path: str):
        for entry in sorted(os.listdir(base_path)):
            class_dir = os.path.join(base_path, entry)
            if not os.path.isdir(class_dir):
                continue

            # 提取失败用例
            extract_bad_cases(class_dir)

            # 生成评测报告
            report_path = os.path.join(class_dir, 'test_report.md')
            md_writer = MarkdownWriter(report_path)

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
        resume=params.resume, agent_classes=params.agent_classes
    ).run()
