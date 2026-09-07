"""逐例异步评价：直接调用 metric.a_measure，不使用 DeepEval evaluate 的全量屏障。"""

import asyncio
import copy
import inspect
import math
from collections import Counter

from tool.config_reader import ConfigReader
from evaluator.metrics import (
    reverse_validation_metric, contextual_recall_metric, mrr_metric,
    recallk_metric, precisionk_metric, dataqa_capability_metric,
    dataqa_params_metric, create_metrics_for_model,
)

METRICS_MAP = {
    'reverse_validation': reverse_validation_metric,
    'contextual_recall': contextual_recall_metric,
    'mrr': mrr_metric, 'recallk': recallk_metric, 'precisionk': precisionk_metric,
    'dataqa_capability': dataqa_capability_metric, 'dataqa_params': dataqa_params_metric,
}
RETRIEVAL_METRICS = {'mrr', 'recallk', 'precisionk'}
STRUCTURED_KEYS = {'dataqa_capability': 'llm_test_case_capability',
                   'dataqa_params': 'llm_test_case_params'}


def recompute_overall_success(test_case: dict):
    """所有指标均为布尔 True 才通过；没有指标结果不能通过。"""
    for case in test_case['csv']:
        keys = [k for k in case if k.endswith('_is_success') and '_model' not in k]
        success = bool(keys) and all(case[k] is True for k in keys)
        case['is_success'] = case['用例是否通过'] = success


def _apply_voting(case: dict, metric_names: list[str]):
    for name in metric_names:
        votes = [case.get(f'{name}{suffix}_is_success')
                 for suffix in ('', '_model2', '_model3')]
        case[f'{name}_is_success'] = sum(v is True for v in votes) >= 2
    recompute_overall_success({'csv': [case]})


def _fresh_metric(template):
    # 保留原生模型类型和客户端；GEval 的原生 logprobs 路径不变。
    # 其余可变状态完全隔离，模板本身从不执行 measure。
    model = getattr(template, 'model', None)
    metric = copy.deepcopy(template, {id(model): model} if model is not None else {})
    for attr in ('score', 'reason', 'success', 'error'):
        setattr(metric, attr, None)
    return metric


class CaseEvaluator:
    """一个数据集共用一个调用配额；每次指标尝试使用独立 metric。"""

    def __init__(self, metrics, conf=None, check_health=lambda: None):
        conf = conf or ConfigReader.get_instance()
        self.metrics = list(dict.fromkeys(metrics))
        for name in self.metrics:
            if name not in METRICS_MAP:
                raise ValueError(f'未注册评价指标: {name}')
        self.concurrency = conf.get('evluate.max_concurrent', 10)
        if type(self.concurrency) is not int or self.concurrency < 1:
            raise ValueError('evluate.max_concurrent 必须为正整数')
        if not conf.get('evluate.run_async', True):
            self.concurrency = 1
        self.inflight = self.concurrency * 2
        self.throttle = conf.get('evluate.throttle_value', 0)
        self.timeout = conf.get('evluate.metric_timeout', 180)
        self.retries = conf.get('retry.eval_max_retries', 2)
        self.backoff = conf.get('retry.backoff_base', 2)
        self.max_backoff = conf.get('retry.max_backoff', 60)
        if type(self.retries) is not int or self.retries < 0:
            raise ValueError('retry.eval_max_retries 必须为非负整数')
        for key, value, minimum in [('throttle_value', self.throttle, 0),
                                    ('metric_timeout', self.timeout, 0.001),
                                    ('backoff_base', self.backoff, 0),
                                    ('max_backoff', self.max_backoff, 0)]:
            if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
                raise ValueError(f'{key} 配置无效')
        self.review_names = [n for n in self.metrics
                             if n in conf.get('metric_conf.needs_review', [])]
        self.review_models = [conf.get(f'judge_llm.anthropic.model{i}') for i in (2, 3)]
        self.templates = {None: {n: METRICS_MAP[n] for n in self.metrics}}
        self.gate = asyncio.Semaphore(self.concurrency)
        self.start_lock = asyncio.Lock()
        self.next_start = 0
        self.check_health = check_health
        self.active = Counter()
        self.retry_waiting = 0

    async def _measure(self, template, test_case, label):
        async with self.gate:
            # 所有轨道、复核和重试使用同一个节流时钟。
            async with self.start_lock:
                self.check_health()
                loop = asyncio.get_running_loop()
                await asyncio.sleep(max(0, self.next_start - loop.time()))
                self.check_health()
                self.next_start = loop.time() + self.throttle
            metric = _fresh_metric(template)
            parameters = inspect.signature(metric.a_measure).parameters
            kwargs = {key: False for key in ('_show_indicator', '_log_metric_to_confident')
                      if key in parameters}
            self.active[label] += 1
            try:
                await asyncio.wait_for(metric.a_measure(test_case, **kwargs), self.timeout)
                score = metric.score
                if (type(score) not in (int, float) or not math.isfinite(score)
                        or getattr(metric, 'error', None)):
                    raise ValueError(getattr(metric, 'error', None) or '指标未返回有效 score')
                return {'score': score, 'is_success': metric.is_successful() is True,
                        'threshold': metric.threshold, 'reason': metric.reason}
            finally:
                self.active[label] -= 1

    async def _metric(self, case, name, template, key, label='primary'):
        prefix = template.__name__ + ('' if label == 'primary' else f'_{label}')
        result = {'score': None, 'is_success': False,
                  'threshold': template.threshold, 'reason': ''}
        error = None
        if key not in case:
            error = f'{key} 缺失，无法评价 {name}'
        else:
            for attempt in range(self.retries + 1):
                self.check_health()
                if attempt:
                    self.retry_waiting += 1
                    try:
                        await asyncio.sleep(min(self.backoff ** attempt, self.max_backoff))
                    finally:
                        self.retry_waiting -= 1
                    self.check_health()
                try:
                    result = await self._measure(template, case[key], label)
                    error = None
                    break
                except Exception as exc:
                    # CancelledError/BaseException 不转成评分失败，不重试取消的调用。
                    self.check_health()
                    error = f'{type(exc).__name__}: {exc}'
                finally:
                    if attempt:
                        case['retry_count'] = max(case.get('retry_count', 0), attempt)
        if error is not None:
            result['reason'] = error
            message = f'[{label}/{name}] {error}'
            previous = case.get('evaluate_error', '')
            case['evaluate_error'] = f'{previous}; {message}' if previous else message
        case.update({f'{prefix}_{field}': value for field, value in result.items()})

    async def evaluate_case(self, case):
        case['is_success'] = False
        for name in self.metrics:
            key = STRUCTURED_KEYS.get(name, 'llm_test_case_retrieval'
                                      if name in RETRIEVAL_METRICS else 'llm_test_case')
            await self._metric(case, name, self.templates[None][name], key)
        recompute_overall_success({'csv': [case]})
        if (case['is_success'] or not self.review_names or not all(self.review_models)
                or 'llm_test_case' not in case):
            return
        reviewed = []
        for index, model in enumerate(self.review_models, 2):
            self.check_health()
            if model not in self.templates:
                self.templates[model] = create_metrics_for_model(model, self.review_names)
            templates = self.templates[model]
            for name, template in templates.items():
                if index == 2:
                    reviewed.append(template.__name__)
                await self._metric(case, name, template, 'llm_test_case', f'model{index}')
        _apply_voting(case, reviewed)

    async def evaluate_group(self, rows):
        # 父用例作为终态单位；子轮依次处理，避免轮数放大在途任务数。
        for row in rows:
            await self.evaluate_case(row)
        return rows


async def evaluate_groups(groups, metrics, on_result, *, conf=None,
                          check_health=lambda: None, write_stats=lambda: {}, label='评价'):
    """有界调度，按完成顺序交付完整父用例。on_result 是唯一结果出口。"""
    evaluator = CaseEvaluator(metrics, conf, check_health)
    completed = asyncio.Queue()
    pending = set()
    delivered = set()
    iterator = iter(groups)
    finished = 0
    started = 0

    def replenish():
        nonlocal started
        while len(pending) < evaluator.inflight:
            check_health()
            rows = next(iterator, None)
            if rows is None:
                break
            task = asyncio.create_task(evaluator.evaluate_group(rows))
            pending.add(task)
            task.add_done_callback(completed.put_nowait)
            started += 1

    async def deliver(task):
        nonlocal finished
        rows = task.result()
        # 回调在首次 await 之前接收 rows；取消时由调用方排空已接收的缓冲。
        delivered.add(task)
        await on_result(rows)
        finished += 1

    async def monitor():
        last_log = 0
        while True:
            check_health()
            now = asyncio.get_running_loop().time()
            if now - last_log >= 1:
                stats = write_stats()
                print(f'[{label}] 终态 {finished}/{len(groups)} | 待启动 {len(groups)-started}'
                      f' | 主评 {evaluator.active["primary"]}'
                      f' | 复核 {evaluator.active["model2"] + evaluator.active["model3"]}'
                      f' | 重试等待 {evaluator.retry_waiting}'
                      f' | 已提交 {stats.get("committed_rows", 0)}'
                      f' | 写入队列 {stats.get("pending", 0)}', flush=True)
                last_log = now
            await asyncio.sleep(0.05)

    async def consume():
        replenish()
        while pending:
            task = await completed.get()
            await deliver(task)
            pending.remove(task)
            delivered.discard(task)
            replenish()

    consumer = asyncio.create_task(consume())
    watcher = asyncio.create_task(monitor())
    try:
        done, _ = await asyncio.wait((consumer, watcher), return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    finally:
        consumer.cancel()
        watcher.cancel()
        await asyncio.gather(consumer, watcher, return_exceptions=True)
        for task in pending:
            if not task.done():
                task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        # 中断时保存已完成但尚未交付的终态；半成品父用例绝不落盘。
        check_health()
        while not completed.empty():
            task = completed.get_nowait()
            if task not in delivered and not task.cancelled() and task.exception() is None:
                await deliver(task)
