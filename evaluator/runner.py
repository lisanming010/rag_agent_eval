"""评测执行器，封装 deepeval evaluate 调用 + 重试 + 结果回写"""

import time

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig

from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory
from evaluator.metrics import (
    reverse_validation_metric,
    contextual_recall_metric,
    mrr_metric,
    recallk_metric,
    precisionk_metric,
    dataqa_capability_metric,
    dataqa_params_metric,
    create_metrics_for_model,
)

logger = LogFactory.get_logger(__name__)

_conf = ConfigReader.get_instance()
METRIC_NEEDS_REVIEW: set[str] = set(_conf.get('metric_conf.needs_review', []))

#指标注册表
METRICS_MAP = {
    'reverse_validation': reverse_validation_metric,
    'contextual_recall': contextual_recall_metric,
    'mrr': mrr_metric,
    'recallk': recallk_metric,
    'precisionk': precisionk_metric,
    'dataqa_capability': dataqa_capability_metric,
    'dataqa_params': dataqa_params_metric,
}


def _resolve_metrics(metrics_str: list[str]) -> list:
    """将指标名称列表解析为 deepeval metric 实例列表"""
    metrics = []
    for metric_name in metrics_str:
        eva_metric = METRICS_MAP.get(metric_name, None)
        if eva_metric is None:
            logger.error(f'{metric_name}尚未注册，请在evaluator/metrics.py中实现后在METRICS_MAP中完成注册')
            raise ValueError(
                f'{metric_name}尚未注册，请在evaluator/metrics.py中实现后在METRICS_MAP中完成注册'
            )
        metrics.append(eva_metric)
    return metrics


# 检索类指标名称 — 需路由到 llm_test_case_retrieval 的指标
RETRIEVAL_METRICS = {'mrr', 'recallk', 'precisionk'}


def _batch_evaluate(cases: list[dict], tc_key: str, metrics: list,
                    track_label: str):
    """单轨评测：evaluate → 重试 → 回写 per-metric 字段

    :param cases: 用例 dict 列表（原地修改）
    :param tc_key: LLMTestCase 在 case dict 中的 key，如 'llm_test_case' 或 'llm_test_case_retrieval'
    :param metrics: deepeval metric 实例列表
    :param track_label: 日志标签，如 '文本质量'、'检索'
    """
    if not metrics:
        return

    conf_reader = ConfigReader.get_instance()

    valid_cases = [c for c in cases if tc_key in c]
    skipped = [c for c in cases if tc_key not in c]

    for case in skipped:
        if 'is_success' not in case:
            case['is_success'] = False
        if 'evaluate_error' not in case:
            case['evaluate_error'] = f'{tc_key}缺失，无法执行{track_label}评测'

    if skipped:
        logger.info(f"[{track_label}] 跳过 {len(skipped)} 条缺少 {tc_key} 的用例")

    if not valid_cases:
        logger.warning(f"[{track_label}] 无有效用例，跳过评测")
        return

    llm_cases = [c[tc_key] for c in valid_cases]

    eval_max_retries = conf_reader.get('retry.eval_max_retries', 2)
    retry_backoff_base = conf_reader.get('retry.backoff_base', 2)
    retry_max_backoff = conf_reader.get('retry.max_backoff', 60)
    retry_verbose = conf_reader.get('retry.verbose', True)

    eva_run_async = conf_reader.get('evluate.run_async')
    eva_max_concurrent = conf_reader.get('evluate.max_concurrent')
    eva_throttle_value = conf_reader.get('evluate.throttle_value')

    # 首次全量 evaluate
    result = evaluate(
        llm_cases,
        metrics,
        async_config=AsyncConfig(
            run_async=eva_run_async,
            max_concurrent=eva_max_concurrent,
            throttle_value=eva_throttle_value
        ),
        error_config=ErrorConfig(ignore_errors=True)
    )

    result_map = {tr.input: tr for tr in result.test_results}

    def _score_missing(tr):
        if not tr.metrics_data:
            return True
        return any(md.score is None for md in tr.metrics_data)

    pending = [
        case for case in valid_cases
        if case['query'] not in result_map
        or _score_missing(result_map[case['query']])
    ]

    # 重试循环
    initial_pending = len(pending)
    retry_round = 0
    while pending and retry_round < eval_max_retries:
        retry_round += 1
        delay = min(retry_backoff_base ** retry_round, retry_max_backoff)

        completed = initial_pending - len(pending)
        if retry_verbose:
            logger.info(f"[{track_label} 重试] 第 {retry_round}/{eval_max_retries} 轮，"
                        f"已完成 {completed}/{initial_pending}，"
                        f"待重试 {len(pending)} 条，等待 {delay:.1f}s")

        time.sleep(delay)

        still_pending = []
        for idx, case in enumerate(pending):
            query = case['query']
            try:
                retry_result = evaluate(
                    [case[tc_key]],
                    metrics,
                    error_config=ErrorConfig(ignore_errors=False)
                )
                if retry_result.test_results:
                    result_map[query] = retry_result.test_results[0]
                    case['retry_count'] = retry_round
                    if retry_verbose:
                        logger.info(f"  ✓ [{completed + idx + 1}/{initial_pending}] "
                                    f"query='{query[:50]}' 第{retry_round}轮重试成功")
                    continue
            except Exception as e:
                if retry_verbose:
                    logger.info(f"  ✗ [{completed + idx + 1}/{initial_pending}] "
                                f"query='{query[:50]}' 重试异常: {e}")

            still_pending.append(case)

        pending = still_pending

    # 最终失败标记
    for case in pending:
        case['is_success'] = False
        case['evaluate_error'] = f"[{track_label}] 经过{eval_max_retries}轮重试后仍无评测结果"
        case['retry_count'] = eval_max_retries
        if retry_verbose:
            logger.error(f"[{track_label} 失败] query='{case['query'][:60]}' "
                         f"经{eval_max_retries}轮重试仍失败")

    # 结果回写（仅 per-metric 字段，is_success 由 recompute_overall_success 统一计算）
    for case_dict in cases:
        query = case_dict['query']
        test_result = result_map.get(query)
        if test_result is None:
            continue

        for md in test_result.metrics_data:
            metrics_name = md.name
            case_dict[f'{metrics_name}_is_success'] = md.success
            case_dict[f'{metrics_name}_score'] = md.score
            case_dict[f'{metrics_name}_threshold'] = md.threshold
            case_dict[f'{metrics_name}_reason'] = md.reason


def run_evaluate(test_case: dict):
    """双轨评测：按 metric 类型分流到文本轨 / 检索轨

    文本轨 → llm_test_case → reverse_validation, contextual_recall 等
    检索轨 → llm_test_case_retrieval → mrr, recallk, precisionk
    """
    test_case_csv = test_case['csv']
    configured = test_case['metrics']

    text_metrics = [m for m in configured if m not in RETRIEVAL_METRICS]
    retrieval_metrics = [m for m in configured if m in RETRIEVAL_METRICS]

    if text_metrics:
        _batch_evaluate(
            test_case_csv, 'llm_test_case',
            _resolve_metrics(text_metrics), '文本质量'
        )

    if retrieval_metrics:
        _batch_evaluate(
            test_case_csv, 'llm_test_case_retrieval',
            _resolve_metrics(retrieval_metrics), '检索'
        )

    # 兜底：完全缺失 LLMTestCase 的用例标记失败
    for case in test_case_csv:
        has_text = 'llm_test_case' in case
        has_retrieval = 'llm_test_case_retrieval' in case
        if not has_text and not has_retrieval:
            if 'is_success' not in case:
                case['is_success'] = False
                case['用例是否通过'] = False
            if 'evaluate_error' not in case:
                case['evaluate_error'] = '所有LLMTestCase均缺失，无法执行评测'


def run_multimodel_reevaluate(test_case: dict):
    """
    对一轮评测后的 bad_cases 使用 model2 和 model3 批量重新评测（仅 LLM 类指标），
    结果追加到 case 字典并按三模型投票更新最终 is_success。

    :param test_case: 一轮评测后的测试数据集 {'csv': [dict], 'case_name': str, 'metrics': [str]}
    """
    conf = ConfigReader.get_instance()
    model2_name = conf.get('judge_llm.anthropic.model2')
    model3_name = conf.get('judge_llm.anthropic.model3')

    if not model2_name or not model3_name:
        logger.warning("未配置 model2/model3，跳过多模型复核")
        return

    test_case_csv = test_case['csv']
    all_metrics = test_case['metrics']

    # 筛选出需要多模型复核的指标（由 config 中 metric_conf.needs_review 控制）
    llm_metrics = [m for m in all_metrics if m in METRIC_NEEDS_REVIEW]
    logger.debug(f'需复核的数据集：\n{llm_metrics}\n')
    if not llm_metrics:
        return

    # 筛选 bad_cases
    bad_cases = [
        case for case in test_case_csv
        if 'llm_test_case' in case and case.get('is_success') is not True
    ]
    if not bad_cases:
        logger.info(f"[多模型复核] 无 bad_case，跳过")
        return

    logger.info(f"[多模型复核] 发现 {len(bad_cases)} 条 bad_case，启动 model2/model3 批量复核")

    # 为 model2/model3 创建 LLM 指标实例
    model2_metrics = create_metrics_for_model(model2_name, llm_metrics)
    model3_metrics = create_metrics_for_model(model3_name, llm_metrics)

    eva_run_async = conf.get('evluate.run_async')
    eva_max_concurrent = conf.get('evluate.max_concurrent')
    eva_throttle_value = conf.get('evluate.throttle_value')

    # 批量评测：model2 和 model3 各一次 evaluate 调用，内部并发
    _batch_reevaluate(bad_cases, model2_metrics, 'model2',
                      eva_run_async, eva_max_concurrent, eva_throttle_value)
    _batch_reevaluate(bad_cases, model3_metrics, 'model3',
                      eva_run_async, eva_max_concurrent, eva_throttle_value)

    # 按计划复核的指标投票，调用失败或无返回结果也必须计入失败票。
    reviewed_metric_names = [metric.__name__ for metric in model2_metrics.values()]
    for case in bad_cases:
        _apply_voting(case, reviewed_metric_names)


def _batch_reevaluate(bad_cases: list[dict], metrics_map: dict, model_label: str,
                      run_async: bool, max_concurrent: int, throttle_value: int):
    """对一批 bad_cases 批量执行评测，内置重试，结果按 query 回写到各自 case 字典"""
    metric_instances = list(metrics_map.values())
    if not metric_instances:
        return

    conf = ConfigReader.get_instance()
    eval_max_retries = conf.get('retry.eval_max_retries', 2)
    retry_backoff_base = conf.get('retry.backoff_base', 2)
    retry_max_backoff = conf.get('retry.max_backoff', 60)

    def _do_evaluate(cases: list[dict], ignore_errors: bool):
        logger.info(f'执行复核，复核模型：{model_label}')
        return evaluate(
            [c['llm_test_case'] for c in cases],
            metric_instances,
            async_config=AsyncConfig(
                run_async=run_async,
                max_concurrent=max_concurrent,
                throttle_value=throttle_value
            ),
            error_config=ErrorConfig(ignore_errors=ignore_errors)
        )

    def _score_missing(tr):
        if not tr.metrics_data:
            return True
        return any(md.score is None for md in tr.metrics_data)

    # 首次批量评测
    try:
        result = _do_evaluate(bad_cases, ignore_errors=True)
    except Exception as e:
        logger.error(f"[多模型复核] {model_label} 批量评测异常: {e}")
        return

    result_map = {tr.input: tr for tr in result.test_results}

    # 筛选 score 缺失的用例（JSON 解析失败等）
    pending = [
        case for case in bad_cases
        if case['query'] not in result_map
        or _score_missing(result_map[case['query']])
    ]

    # 重试循环
    initial_pending = len(pending)
    retry_round = 0
    while pending and retry_round < eval_max_retries:
        retry_round += 1
        delay = min(retry_backoff_base ** retry_round, retry_max_backoff)
        completed = initial_pending - len(pending)
        logger.info(f"[多模型复核] {model_label} 第 {retry_round}/{eval_max_retries} 轮重试，"
                    f"已完成 {completed}/{initial_pending}，"
                    f"待重试 {len(pending)} 条，等待 {delay:.1f}s")
        time.sleep(delay)

        still_pending = []
        for idx, case in enumerate(pending):
            try:
                retry_result = _do_evaluate([case], ignore_errors=False)
                if retry_result.test_results and not _score_missing(retry_result.test_results[0]):
                    result_map[case['query']] = retry_result.test_results[0]
                    logger.info(f"  ✓ [{completed + idx + 1}/{initial_pending}] "
                                f"query='{case['query'][:50]}' {model_label}重试成功")
                    continue
            except Exception as e:
                logger.info(f"  ✗ [{completed + idx + 1}/{initial_pending}] "
                            f"query='{case['query'][:50]}' {model_label}重试异常: {e}")
            still_pending.append(case)

        pending = still_pending

    if pending:
        logger.warning(f"[多模型复核] {model_label} {len(pending)} 条经{eval_max_retries}轮重试仍无结果")

    # 结果回写（按 query 索引，异步模式下顺序可能不一致）
    for case in bad_cases:
        tr = result_map.get(case['query'])
        if tr is None:
            continue
        for md in tr.metrics_data:
            case[f'{md.name}_{model_label}_score'] = md.score
            case[f'{md.name}_{model_label}_is_success'] = md.success
            case[f'{md.name}_{model_label}_reason'] = md.reason
            case[f'{md.name}_{model_label}_threshold'] = md.threshold


def _apply_voting(case: dict, metric_names: list[str]):
    """
    基于三模型投票更新 case 的最终 is_success：
    对每个计划复核的指标，仅布尔 True 算通过，三模型中至少两个通过才最终通过。
    缺失、None 和其他非 True 值均计为失败票。
    整体 is_success = 所有指标的最终结果取 AND。
    """
    for metric_prefix in metric_names:
        m1 = case.get(f'{metric_prefix}_is_success')
        m2 = case.get(f'{metric_prefix}_model2_is_success')
        m3 = case.get(f'{metric_prefix}_model3_is_success')

        pass_count = sum(1 for v in (m1, m2, m3) if v is True)
        case[f'{metric_prefix}_is_success'] = pass_count >= 2

    # 重新计算整体 is_success：所有指标 _is_success 取 AND
    all_success = True
    for key, val in case.items():
        if key.endswith('_is_success') and key != 'is_success' and '_model' not in key:
            if val is not True:
                all_success = False
                break
    case['is_success'] = all_success
    # 反向归一化：is_success → 用例是否通过
    case['用例是否通过'] = all_success


def run_evaluate_structured(test_case: dict):
    """执行 DataQA 结构化断言（capability_id + parameters）。

    仅写入 per-metric 字段，不修改 is_success。
    """
    configured = test_case['metrics']

    if 'dataqa_capability' in configured:
        _eval_single_structured_metric(
            test_case['csv'], dataqa_capability_metric, 'llm_test_case_capability'
        )

    if 'dataqa_params' in configured:
        _eval_single_structured_metric(
            test_case['csv'], dataqa_params_metric, 'llm_test_case_params'
        )


def _eval_single_structured_metric(csv_rows: list[dict], metric, tc_key: str):
    """对单个结构化 metric 执行 evaluate 并回写 per-metric 字段。"""
    valid_cases = [c for c in csv_rows if tc_key in c]
    if not valid_cases:
        logger.debug(f"[结构化] {metric.__name__}: 无有效用例（缺少 {tc_key}），跳过")
        return

    try:
        result = evaluate(
            [c[tc_key] for c in valid_cases],
            [metric],
            error_config=ErrorConfig(ignore_errors=True),
        )
    except Exception as e:
        logger.error(f"[结构化] {metric.__name__} evaluate 异常: {e}")
        return

    result_map = {tr.input: tr for tr in result.test_results}

    for case in csv_rows:
        tr = result_map.get(case['query'])
        if tr is None:
            continue
        for md in tr.metrics_data:
            case[f'{md.name}_score'] = md.score
            case[f'{md.name}_is_success'] = md.success
            case[f'{md.name}_reason'] = md.reason
            case[f'{md.name}_threshold'] = md.threshold

    ok = sum(1 for c in valid_cases
             if result_map.get(c['query']) and result_map[c['query']].success)
    logger.info(f"[结构化] {metric.__name__}: {ok}/{len(valid_cases)} 通过")


def recompute_overall_success(test_case: dict):
    """所有 *_is_success 字段（排除 _model）均为布尔 True 才通过。"""
    for case in test_case['csv']:
        metric_keys = [
            k for k in case
            if k.endswith('_is_success')
            and k != 'is_success'
            and '_model' not in k
        ]
        if not metric_keys:
            continue
        all_pass = all(case[k] is True for k in metric_keys)
        case['is_success'] = all_pass
        case['用例是否通过'] = all_pass
