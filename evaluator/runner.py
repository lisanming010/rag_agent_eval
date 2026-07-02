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
    create_metrics_for_model,
)

logger = LogFactory.get_logger(__name__)

#指标注册表 
METRICS_MAP = {
    'reverse_validation': reverse_validation_metric,
    'contextual_recall': contextual_recall_metric,
    'mrr': mrr_metric,
    'recallk': recallk_metric,
    'precisionk': precisionk_metric,
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


def run_evaluate(test_case: dict):
    """
    执行evaluate，内置重试机制，并将执行结果写回用例字典中

    :param test_case: 测试数据集, 即测试用例list中的整个成员，如 test_case_list[0]
    """

    conf_reader = ConfigReader.get_instance()

    test_case_csv = test_case['csv']

    # 分离出缺少 llm_test_case 的用例（make_llm_case 阶段已失败）
    valid_cases = [case for case in test_case_csv if 'llm_test_case' in case]
    skipped_cases = [case for case in test_case_csv if 'llm_test_case' not in case]

    if skipped_cases:
        for case in skipped_cases:
            if 'is_success' not in case:
                case['is_success'] = False
            if 'evaluate_error' not in case:
                case['evaluate_error'] = 'llm_test_case缺失，无法执行评测'
        logger.info(f"[跳过] {len(skipped_cases)} 条用例缺少 llm_test_case，已标记为失败")

    if not valid_cases:
        logger.error("[中止] 没有可评测的有效用例")
        return

    # 提取字典中所有 llm_test_case
    llm_test_case = [case['llm_test_case'] for case in valid_cases]
    metrics = _resolve_metrics(test_case['metrics'])

    #读取 retry 配置
    eval_max_retries = conf_reader.get('retry.eval_max_retries', 2)
    retry_backoff_base = conf_reader.get('retry.backoff_base', 2)
    retry_max_backoff = conf_reader.get('retry.max_backoff', 60)
    retry_verbose = conf_reader.get('retry.verbose', True)

    eva_run_async = conf_reader.get('evluate.run_async')
    eva_max_concurrent = conf_reader.get('evluate.max_concurrent')
    eva_throttle_value = conf_reader.get('evluate.throttle_value')

    #首次全量 evaluate
    result = evaluate(
        llm_test_case,
        metrics,
        async_config=AsyncConfig(
            run_async=eva_run_async,
            max_concurrent=eva_max_concurrent,
            throttle_value=eva_throttle_value
        ),
        error_config=ErrorConfig(ignore_errors=True)
    )

    # 异步模式下 deepeval 不保证 test_results 顺序与输入一致，
    # 因此按 test_result.input（原始query）建立索引后再回写，避免张冠李戴
    result_map = {tr.input: tr for tr in result.test_results}

    # 收集因 LLM 报错被跳过的用例（仅从有效用例中查找）
    # - query 不在 result_map 中：deepeval 直接丢弃了该用例
    # - query 在 result_map 但 metrics_data 为空：评测未执行完就报错
    # 注意：metrics_data 非空但 success=False 是正常低分，不重试
    # 检查 metric score 是否缺失：空列表或任一 score 为 None 均视为未完整执行
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
    retry_round = 0
    while pending and retry_round < eval_max_retries:
        retry_round += 1
        delay = min(retry_backoff_base ** retry_round, retry_max_backoff)

        if retry_verbose:
            logger.info(f"[重试] 第 {retry_round}/{eval_max_retries} 轮，"
                        f"待重试用例: {len(pending)} 条，等待 {delay:.1f}s")

        time.sleep(delay)

        still_pending = []
        for idx, case in enumerate(pending):
            query = case['query']
            try:
                # 单用例重试时关闭 ignore_errors，让异常暴露出来
                retry_result = evaluate(
                    [case['llm_test_case']],
                    metrics,
                    error_config=ErrorConfig(ignore_errors=False)
                )
                if retry_result.test_results:
                    result_map[query] = retry_result.test_results[0]
                    case['retry_count'] = retry_round
                    if retry_verbose:
                        logger.info(f"  ✓ [{idx+1}/{len(pending)}] "
                                    f"query='{query[:50]}' 第{retry_round}轮重试成功")
                    continue
            except Exception as e:
                if retry_verbose:
                    logger.info(f"  ✗ [{idx+1}/{len(pending)}] "
                                f"query='{query[:50]}' 重试异常: {e}")

            still_pending.append(case)

        pending = still_pending

    #最终失败标记
    for case in pending:
        case['is_success'] = False
        case['evaluate_error'] = f"经过{eval_max_retries}轮重试后仍无评测结果"
        case['retry_count'] = eval_max_retries
        if retry_verbose:
            logger.error(f"[失败] query='{case['query'][:60]}' 经{eval_max_retries}轮重试仍失败")

    # 结果回写 
    for case_dict in test_case_csv:
        query = case_dict['query']
        test_result = result_map.get(query)
        if test_result is None:
            # 已在上面标记过 evaluate_error，这里仅补充 is_success 兜底
            if 'is_success' not in case_dict:
                case_dict['is_success'] = False
            continue

        case_dict['is_success'] = test_result.success
        for md in test_result.metrics_data:
            metrics_name = md.name
            case_dict[f'{metrics_name}_is_success'] = md.success
            case_dict[f'{metrics_name}_score'] = md.score
            case_dict[f'{metrics_name}_threshold'] = md.threshold
            case_dict[f'{metrics_name}_reason'] = md.reason


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

    # 筛选出 LLM 类指标（只有这些需要多模型复核）
    llm_metrics = [m for m in all_metrics if m in ('reverse_validation', 'contextual_recall')]
    logger.debug(f'需复核的数据集：\n{llm_metrics}\n')
    if not llm_metrics:
        return

    # 筛选 bad_cases
    bad_cases = [
        case for case in test_case_csv
        if 'llm_test_case' in case and case.get('is_success') != True
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

    # 逐条投票
    for case in bad_cases:
        _apply_voting(case)


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
    retry_round = 0
    while pending and retry_round < eval_max_retries:
        retry_round += 1
        delay = min(retry_backoff_base ** retry_round, retry_max_backoff)
        logger.info(f"[多模型复核] {model_label} 第 {retry_round}/{eval_max_retries} 轮重试，"
                    f"待重试: {len(pending)} 条，等待 {delay:.1f}s")
        time.sleep(delay)

        still_pending = []
        for idx, case in enumerate(pending):
            try:
                retry_result = _do_evaluate([case], ignore_errors=False)
                if retry_result.test_results and not _score_missing(retry_result.test_results[0]):
                    result_map[case['query']] = retry_result.test_results[0]
                    logger.info(f"  ✓ [{idx+1}/{len(pending)}] "
                                f"query='{case['query'][:50]}' {model_label}重试成功")
                    continue
            except Exception as e:
                logger.info(f"  ✗ [{idx+1}/{len(pending)}] "
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


def _apply_voting(case: dict):
    """
    基于三模型投票更新 case 的最终 is_success：
    对每个被复核过的指标，若 model1/model2/model3 中 ≥2 个判定不通过则最终不通过。
    整体 is_success = 所有指标的最终结果取 AND。
    """
    # 找到所有被 model2 复核过的指标列
    model2_success_keys = [k for k in case if k.endswith('_model2_is_success')]

    for m2_key in model2_success_keys:
        # contextual_recall_model2_is_success → contextual_recall
        metric_prefix = m2_key.removesuffix('_model2_is_success')

        m1 = case.get(f'{metric_prefix}_is_success', True)
        m2 = case.get(f'{metric_prefix}_model2_is_success', True)
        m3 = case.get(f'{metric_prefix}_model3_is_success', True)

        fail_count = sum(1 for v in (m1, m2, m3) if not v)
        case[f'{metric_prefix}_is_success'] = fail_count < 2

    # 重新计算整体 is_success：所有指标 _is_success 取 AND
    all_success = True
    for key, val in case.items():
        if key.endswith('_is_success') and key != 'is_success' and '_model' not in key:
            if not val:
                all_success = False
                break
    case['is_success'] = all_success
