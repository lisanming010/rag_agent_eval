"""恢复评价的只读预检、历史筛选和 tmp 多轮还原；不创建 Agent 或评价模型。"""

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from tool.result_checkpoint import csv_value, read_committed
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)
CORE_FIELDS = ('query', 'agent_response', 'expected_behavior', 'res_time(s)')
INPUT_FIELDS = (
    'query', 'agent_response', 'expected_behavior', 'forbidden_behavior',
    'actual_capability_id', 'actual_parameters', 'expected_capability_id',
    'expected_parameters', 'expected_docs', 'retrieved_docs', 'skip_assertion',
)
RETRIEVAL_METRICS = {'mrr', 'recallk', 'precisionk'}


def basename(value) -> str:
    return str(value).replace('\\', '/').rsplit('/', 1)[-1]


def result_filename(case_name: str) -> str:
    return basename(case_name).replace('test_case', 'result_output')


def source_case_name(tmp_path: Path, meta: dict | None, rows: list[dict]) -> str:
    """原始路径可移动，但同一 tmp 的来源文件名必须唯一且与标准名称一致。"""
    inferred = tmp_path.name.removesuffix('_tmp.csv') + '.csv'
    candidates = [row['_source_csv'] for row in rows if row.get('_source_csv')]
    if meta and meta.get('case_name'):
        candidates.insert(0, meta['case_name'])
    names = {basename(name).casefold() for name in candidates}
    names.add(inferred.casefold())
    if len(names) != 1 or not inferred.startswith('test_cases_'):
        raise ValueError(f'tmp 数据集来源不一致或非标准命名: {tmp_path}')
    return str(candidates[0]) if candidates else str(tmp_path.parent.parent / inferred)


def _is_true(value) -> bool:
    return value is True or (isinstance(value, str) and value.strip().casefold() == 'true')


def _turn_count(row: dict) -> int:
    if not row.get('_parent_case_id'):
        return 1
    try:
        count = int(row['_total_turns'])
    except (KeyError, ValueError, TypeError) as exc:
        raise ValueError('多轮 tmp 缺少有效的 _total_turns') from exc
    if count < 1:
        raise ValueError('多轮 tmp 的 _total_turns 必须大于零')
    return count


def _turn_value(row: dict, key: str, turn: int = 1):
    if row.get('_parent_case_id'):
        if key in CORE_FIELDS:
            return row.get(f'{key}(t{turn})', '')
        return row.get(f'{key}(t{turn})', row.get(key, ''))
    value = row.get(key)
    return row.get(f'{key}(t1)', '') if value in (None, '') else value


def _identity(row: dict):
    language = csv_value(row.get('language')).strip().casefold()
    for field in ('test_id', '用例编号', 'case_id', 'id', '_parent_case_id'):
        if csv_value(row.get(field)).strip():
            return 'id', csv_value(row[field]).strip(), language
    queries = tuple(csv_value(_turn_value(row, 'query', i)) for i in range(1, _turn_count(row) + 1))
    if not any(queries):
        raise ValueError('用例缺少稳定编号和 query，无法对应历史结果')
    return 'query', queries, language


def _inputs(row: dict):
    return tuple(tuple(csv_value(_turn_value(row, field, i)) for field in INPUT_FIELDS)
                 for i in range(1, _turn_count(row) + 1))


def _clean_row(row: dict) -> dict:
    """重评只使用 tmp 输入，清除可能残留的旧评价字段，不合并历史失败结论。"""
    clean = {}
    for key, value in row.items():
        base = re.sub(r'\(t\d+\)$', '', key)
        if (base in {'is_success', '用例是否通过', '_parent_all_pass', 'evaluate_error',
                     'retry_count', 'resume_skip_reason'}
                or base.endswith(('_score', '_is_success', '_reason', '_threshold'))
                or base.startswith('llm_test_case')
                or re.fullmatch(r'第\d+轮对话是否通过', base)):
            continue
        clean[key] = value
    return clean


def _missing_responses(row: dict, agent: str, metrics: list[str]) -> list[tuple[int, str]]:
    missing = []
    for turn in range(1, _turn_count(row) + 1):
        if _is_true(row.get('skip_assertion')):
            reason = 'Agent 响应不满足断言前置条件（skip_assertion）'
        elif not csv_value(_turn_value(row, 'query', turn)).strip():
            reason = 'query 为空'
        elif agent.casefold() == 'ragflowretriever':
            reason = '' if csv_value(_turn_value(row, 'retrieved_docs', turn)).strip() else 'retrieved_docs 缺失'
        elif agent.casefold() == 'dataqa' or any(row.get(k) not in (None, '') for k in ('actual_capability_id', 'actual_parameters')):
            absent = [k for k in ('actual_capability_id', 'actual_parameters')
                      if not csv_value(_turn_value(row, k, turn)).strip()]
            reason = '缺少 ' + ', '.join(absent) if absent else ''
        else:
            reason = '' if csv_value(_turn_value(row, 'agent_response', turn)).strip() else 'agent_response 缺失或为空'
        if not reason and set(metrics) & RETRIEVAL_METRICS:
            if not csv_value(_turn_value(row, 'retrieved_docs', turn)).strip():
                reason = 'retrieved_docs 缺失或为空'
        if reason:
            missing.append((turn, reason))
    return missing


def _unavailable(row: dict, missing: list[tuple[int, str]]) -> dict:
    result = _clean_row(row)
    result['is_success'] = False
    result['用例是否通过'] = False
    result['resume_skip_reason'] = '；'.join(f'第 {turn} 轮：{reason}' for turn, reason in missing)
    if row.get('_parent_case_id'):
        result['_parent_all_pass'] = False
        for turn, reason in missing:
            result[f'evaluate_error(t{turn})'] = reason
    else:
        result['evaluate_error'] = result['resume_skip_reason']
    return result


def _expand(row: dict, metrics: list[str]) -> list[dict]:
    clean = _clean_row(row)
    count = _turn_count(row)
    multi = bool(row.get('_parent_case_id'))
    children = []
    for turn in range(1, count + 1):
        child = {key: value for key, value in clean.items() if not re.search(r'\(t\d+\)$', key)}
        for field in CORE_FIELDS:
            child[field] = _turn_value(clean, field, turn)
        if multi:
            child.update(_turn=turn, _total_turns=count, _is_multi_turn_sub=True)
        # 字符串 "False" 不能被当作真值；检索轨按本次有效 metrics 恢复。
        child['_need_retrieval'] = bool(set(metrics) & RETRIEVAL_METRICS)
        if child['_need_retrieval'] and isinstance(child.get('retrieved_docs'), str):
            try:
                docs = ast.literal_eval(child['retrieved_docs'])
            except (ValueError, SyntaxError) as exc:
                raise ValueError('tmp 的 retrieved_docs 不是有效列表') from exc
            if not isinstance(docs, list) or any(not isinstance(doc, str) for doc in docs):
                raise ValueError('tmp 的 retrieved_docs 必须为字符串列表')
            child['retrieved_docs'] = docs
        children.append(child)
    return children


def _matching_child(directory: Path, name: str) -> Path:
    if not directory.exists():
        return directory / name
    if not directory.is_dir():
        raise ValueError(f'历史 Agent 路径不是目录: {directory}')
    matches = [path for path in directory.iterdir() if path.name.casefold() == name.casefold()]
    if len(matches) > 1:
        raise ValueError(f'历史路径存在大小写歧义: {directory / name}')
    return matches[0] if matches else directory / name


@dataclass
class ResumeSummary:
    total: int = 0
    reused: int = 0
    reevaluate: int = 0
    unavailable: int = 0

    @property
    def all_passed(self):
        return self.total > 0 and self.total == self.reused


def prepare_resume(cases_by_agent: dict, result_dir: str | None = None) -> tuple[dict, ResumeSummary]:
    """完整预检后返回新用例组；调用方此后才能组装评价用例、创建输出。"""
    root = Path(result_dir) if result_dir is not None else None
    if result_dir is not None and not str(result_dir).strip():
        raise ValueError('--resume-result-dir 不能为空')
    if root is not None and not root.is_dir():
        raise ValueError(f'--resume-result-dir 不存在或不是目录: {root}')
    prepared = {}
    summary = ResumeSummary()
    for agent, datasets in cases_by_agent.items():
        agent_dir = _matching_child(root, agent.casefold()) if root is not None else None
        names = set()
        prepared[agent] = []
        for dataset in datasets:
            output_name = result_filename(dataset['case_name'])
            if output_name.casefold() in names:
                raise ValueError(f'同一 Agent 的多个 tmp 对应同一结果文件: {agent}/{output_name}')
            names.add(output_name.casefold())
            history = []
            if agent_dir is not None:
                output = _matching_child(agent_dir, output_name)
                snapshot = read_committed(output, agent)
                history = snapshot.rows
                if snapshot.has_uncommitted_tail:
                    logger.warning(f'[resume] 不采纳未提交尾部: {output}')
            index = {}
            candidates = []
            current_keys, current_ids = set(), set()
            for row in dataset['csv']:
                missing = _missing_responses(row, agent, dataset.get('metrics', []))
                try:
                    key = _identity(row)
                except ValueError:
                    if not missing:
                        raise
                    key = None
                if root is not None and key is not None and key[0] == 'id':
                    if key in current_ids:
                        raise ValueError(f'tmp 用例编号有歧义: {agent}/{output_name}: {key[1]}')
                    current_ids.add(key)
                if not missing:
                    current_keys.add(key)
                candidates.append((row, missing, key))
            for old_row in history:
                try:
                    key = _identity(old_row)
                except ValueError:
                    # 无法对应本次有效输入的孤立历史行不参与筛选。
                    continue
                if key not in current_keys:
                    continue
                if key[0] == 'id' and key in index:
                    raise ValueError(f'历史用例编号有歧义: {agent}/{output_name}: {key[1]}')
                # query 重复问题按约定排除，不增加重复检测。
                index[key] = old_row
            pending, reused, skipped = [], [], []
            for row, missing, key in candidates:
                summary.total += 1
                if missing:
                    skipped.append(_unavailable(row, missing))
                    summary.unavailable += 1
                    continue
                old = index.get(key) if root is not None else None
                if old is not None and _inputs(row) == _inputs(old) and _is_true(old.get('is_success')):
                    reused.append(dict(old))
                    summary.reused += 1
                else:
                    pending.extend(_expand(row, dataset.get('metrics', [])))
                    summary.reevaluate += 1
            prepared[agent].append({
                **dataset, 'csv': pending, 'reused_rows': reused, 'skipped_rows': skipped,
                'resume_has_multi_turn': (any(row.get('_parent_case_id') for row in dataset['csv'])
                                         or any('query(t1)' in row for row in history)),
            })
    if not summary.total:
        raise ValueError('tmp 中没有可恢复的用例记录')
    return prepared, summary
