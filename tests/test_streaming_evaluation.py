"""真实调度器 + 模拟 metric + 临时 CSV/checkpoint；不访问评价服务。"""

import asyncio
import threading
from collections import Counter
from types import SimpleNamespace

import pytest

import evaluator.runner as runner
import main
from pipeline.resume import prepare_resume
from tool.async_result_writer import AsyncResultWriter
from tool.result_checkpoint import CheckpointWriter, read_committed


class Config:
    def __init__(self, **values):
        self.values = {'evluate.max_concurrent': 2, 'evluate.throttle_value': 0,
                       'retry.eval_max_retries': 0, 'retry.backoff_base': 0.001,
                       'metric_conf.needs_review': ['reverse_validation'],
                       'judge_llm.anthropic.model2': 'model2',
                       'judge_llm.anthropic.model3': 'model3', **values}

    def get(self, key, default=None):
        return self.values.get(key, default)


class Metric:
    __name__ = 'Judge'
    threshold = 0.7

    def __init__(self, hook, label='primary'):
        self.hook, self.label = hook, label
        self.score = self.success = self.reason = None
        self.history = []

    async def a_measure(self, case, _show_indicator=True, _log_metric_to_confident=True):
        assert not _show_indicator and not _log_metric_to_confident
        assert not self.history  # 每次尝试均是新的指标，不共享 mutable state。
        self.history.append(case.input)
        verdict = await self.hook(case.input, self.label)
        self.score = None if verdict is None else (1 if verdict is True else 0)
        self.success = verdict
        self.reason = f'{self.label}:{case.input}'

    def is_successful(self):
        return self.success


def install(monkeypatch, hook):
    template = Metric(hook)
    monkeypatch.setitem(runner.METRICS_MAP, 'reverse_validation', template)
    monkeypatch.setattr(runner, 'create_metrics_for_model',
                        lambda model, names: {n: Metric(hook, model) for n in names})
    return template


def row(name):
    return {'test_id': str(name), 'query': str(name), 'agent_response': 'answer',
            'expected_behavior': 'expected', 'language': 'en-US',
            'llm_test_case': SimpleNamespace(input=str(name))}


def run_pipeline(tmp_path, monkeypatch, rows, **config):
    instance = main.EvaluationPipeline.__new__(main.EvaluationPipeline)
    instance.conf = Config(**{'result.save_path': str(tmp_path), **config})
    monkeypatch.setattr(main, 'mkdir_with_timestamp', lambda path: path)
    dataset = {'case_name': 'test_cases_demo.csv', 'metrics': ['reverse_validation'], 'csv': rows}
    instance._evaluate({'Diagnosis': [dataset]})
    return dataset, tmp_path / 'diagnosis' / 'result_outputs_demo.csv'


def test_own_review_starts_before_all_primary_and_fast_result_does_not_wait(monkeypatch):
    async def scenario():
        slow_primary, slow_review = asyncio.Event(), asyncio.Event()
        review_started, fast_done = asyncio.Event(), asyncio.Event()
        events, outputs = [], []

        async def hook(query, label):
            events.append((query, label))
            if query == 'slow' and label == 'primary':
                await slow_primary.wait()
            if query == 'bad' and label == 'model2':
                review_started.set()
                await slow_review.wait()
            return not (query == 'bad' and label == 'primary')

        template = install(monkeypatch, hook)

        async def receive(rows):
            outputs.append(rows[0])
            if rows[0]['query'] == 'fast':
                fast_done.set()

        task = asyncio.create_task(runner.evaluate_groups(
            [[row(q)] for q in ('bad', 'slow', 'fast')], ['reverse_validation'], receive,
            conf=Config(**{'evluate.max_concurrent': 3})))
        await asyncio.wait_for(review_started.wait(), 2)
        await asyncio.wait_for(fast_done.wait(), 2)
        assert [r['query'] for r in outputs] == ['fast']
        assert ('bad', 'model3') not in events
        slow_review.set()
        slow_primary.set()
        await asyncio.wait_for(task, 2)
        assert ('fast', 'model2') not in events
        assert all(r['is_success'] is True for r in outputs)
        assert template.history == []
        assert template.score is None

    asyncio.run(scenario())


def test_shared_limit_and_retry_releases_quota(monkeypatch):
    async def scenario():
        calls, active, peak = Counter(), 0, 0
        events = []

        async def hook(query, label):
            nonlocal active, peak
            active += 1
            peak = max(active, peak)
            calls[query, label] += 1
            events.append((query, label, calls[query, label]))
            try:
                await asyncio.sleep(0.002)
                if query == '0' and label == 'primary' and calls[query, label] == 1:
                    raise RuntimeError('retry me')
                return label != 'primary'
            finally:
                active -= 1

        install(monkeypatch, hook)
        outputs = []

        async def receive(rows):
            outputs.extend(rows)

        await runner.evaluate_groups([[row(i)] for i in range(12)], ['reverse_validation'], receive,
                                     conf=Config(**{'retry.eval_max_retries': 1,
                                                    'retry.backoff_base': 0.03}))
        assert peak == 2 and active == 0
        assert events.index(('1', 'primary', 1)) < events.index(('0', 'primary', 2))
        assert len(outputs) == 12 and all(r['is_success'] for r in outputs)
        assert next(r for r in outputs if r['query'] == '0')['retry_count'] == 1

    asyncio.run(scenario())


@pytest.mark.parametrize('failure', ['exception', 'missing_score', 'timeout', 'non_boolean'])
def test_review_failure_is_failed_vote(monkeypatch, failure):
    calls = []

    async def hook(query, label):
        calls.append(label)
        if label == 'model3':
            if failure == 'exception':
                raise TypeError('metric internal TypeError, must not double call')
            if failure == 'timeout':
                await asyncio.sleep(10)
            return None if failure == 'missing_score' else 'True'
        return label == 'model2'

    install(monkeypatch, hook)
    case = row('bad')
    evaluator = runner.CaseEvaluator(['reverse_validation'], Config(**{'evluate.metric_timeout': 0.02}))
    asyncio.run(evaluator.evaluate_case(case))
    assert calls == ['primary', 'model2', 'model3']
    assert case['Judge_model2_is_success'] is True
    assert case['Judge_model3_is_success'] is False
    assert case['is_success'] is False


def test_structured_and_retrieval_route_once_and_missing_metric_cannot_pass(monkeypatch):
    seen = []

    class SimpleMetric:
        threshold = 1

        def __init__(self, name):
            self.__name__ = name

        async def a_measure(self, case):
            seen.append(case.input)
            self.score, self.success, self.reason = 1, True, 'ok'

        def is_successful(self):
            return self.success

    for name in ('mrr', 'dataqa_capability', 'dataqa_params'):
        monkeypatch.setitem(runner.METRICS_MAP, name, SimpleMetric(name))
    case = {'llm_test_case_retrieval': SimpleNamespace(input='retrieval'),
            'llm_test_case_capability': SimpleNamespace(input='capability')}
    evaluator = runner.CaseEvaluator(['mrr', 'dataqa_capability', 'dataqa_params'], Config())
    asyncio.run(evaluator.evaluate_case(case))
    assert seen == ['retrieval', 'capability']
    assert case['mrr_is_success'] is True
    assert case['dataqa_params_is_success'] is False
    assert case['is_success'] is False


def test_first_checkpoint_written_while_slow_primary_is_pending(tmp_path, monkeypatch):
    committed = threading.Event()
    original = CheckpointWriter.commit
    batches = []

    def commit(self, rows):
        original(self, rows)
        batches.append([r['query'] for r in rows])
        committed.set()

    monkeypatch.setattr(CheckpointWriter, 'commit', commit)

    async def hook(query, label):
        if query == '0':
            async with asyncio.timeout(3):
                while not committed.is_set():
                    await asyncio.sleep(0.005)
        return True

    install(monkeypatch, hook)
    rows = [row(i) for i in range(41)]
    inputs = {'Diagnosis': [{'case_name': 'test_cases_demo.csv', 'metrics': ['reverse_validation'],
                            'csv': [dict(r) for r in rows]}]}
    dataset, path = run_pipeline(tmp_path, monkeypatch, rows)
    assert len(batches[0]) == 20 and '0' not in batches[0]
    saved = read_committed(path, 'Diagnosis').rows
    assert len(saved) == len(dataset['csv']) == 41
    assert {r['test_id'] for r in saved} == {str(i) for i in range(41)}
    assert prepare_resume(inputs, str(tmp_path))[1].all_passed


def test_slow_disk_does_not_block_loop_and_backpressure_is_bounded(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = CheckpointWriter.commit
    measured = []

    def commit(self, rows):
        entered.set()
        assert release.wait(4)
        original(self, rows)

    monkeypatch.setattr(CheckpointWriter, 'commit', commit)

    async def hook(query, label):
        measured.append(query)
        await asyncio.sleep(0)
        return True

    install(monkeypatch, hook)

    async def scenario():
        task = asyncio.create_task(asyncio.to_thread(
            run_pipeline, tmp_path, monkeypatch, [row(i) for i in range(200)]))
        try:
            async with asyncio.timeout(3):
                while not entered.is_set() or len(measured) < 45:
                    await asyncio.sleep(0.005)
            await asyncio.sleep(0.1)
            # 1 正写批次 + 2 等待批次 + 1 缓冲批次 + 最多 2*C 在途父用例。
            assert 45 <= len(measured) <= 84
        finally:
            release.set()
        await asyncio.wait_for(task, 4)

    asyncio.run(scenario())


def test_write_failure_cancels_inflight_and_no_new_metric_starts(tmp_path, monkeypatch):
    failed = threading.Event()
    calls = []

    def commit(self, rows):
        failed.set()
        raise OSError('disk unavailable')

    monkeypatch.setattr(CheckpointWriter, 'commit', commit)

    async def hook(query, label):
        assert not failed.is_set()
        calls.append(query)
        await asyncio.sleep(0.001)
        return True

    install(monkeypatch, hook)
    with pytest.raises(RuntimeError, match='disk unavailable'):
        run_pipeline(tmp_path, monkeypatch, [row(i) for i in range(200)])
    assert 20 <= len(calls) < 200
    assert not any(t.name == 'AsyncResultWriter' for t in threading.enumerate())


def test_cancellation_flushes_only_complete_parents_and_partial_batch(tmp_path, monkeypatch):
    async def hook(query, label):
        if query == 'cancel':
            await asyncio.sleep(0.03)
            raise asyncio.CancelledError()
        return True

    install(monkeypatch, hook)
    rows = [row(i) for i in range(13)]
    rows += [{**row('turn1'), '_parent_case_id': 'parent', '_turn': 1},
             {**row('cancel'), '_parent_case_id': 'parent', '_turn': 2}]
    with pytest.raises(asyncio.CancelledError):
        run_pipeline(tmp_path, monkeypatch, rows)
    saved = read_committed(tmp_path / 'diagnosis' / 'result_outputs_demo.csv', 'Diagnosis').rows
    assert len(saved) == 13
    assert all(not r.get('_parent_case_id') for r in saved)


def test_async_writer_freezes_nested_values(tmp_path):
    writer = AsyncResultWriter(tmp_path)
    writer.start()
    values = ['original']
    try:
        asyncio.run(writer.submit_async({'csv': [{'query': 'q', 'nested': values}]}, 'demo.csv'))
        values.append('mutated')
        writer.flush()
        assert read_committed(tmp_path / 'demo.csv', tmp_path.name).rows[0]['nested'] == "['original']"
    finally:
        writer.wait_and_stop()


def test_native_metric_clone_preserves_model_type_and_isolates_steps():
    template = runner.reverse_validation_metric
    clone = runner._fresh_metric(template)
    assert clone.model is template.model
    assert clone.using_native_model == template.using_native_model
    assert clone.evaluation_steps == template.evaluation_steps
    assert clone.evaluation_steps is not template.evaluation_steps
    clone.evaluation_steps.append('not shared')
    assert clone.evaluation_steps != template.evaluation_steps


@pytest.mark.parametrize('initial,llm_pass,structured_pass,expected_review', [
    (None, True, True, False), (None, False, True, True),
    (None, True, False, True), (True, False, True, True),
    (False, True, True, False),
])
def test_review_decision_uses_all_primary_metrics_not_stale_flag(
    monkeypatch, initial, llm_pass, structured_pass, expected_review,
):
    calls = []

    async def hook(query, label):
        calls.append(label)
        return llm_pass if label == 'primary' else True

    install(monkeypatch, hook)

    async def structured(query, label):
        return structured_pass

    metric = Metric(structured)
    metric.__name__ = 'Structured'
    monkeypatch.setitem(runner.METRICS_MAP, 'dataqa_capability', metric)
    case = {**row('q'), 'is_success': initial,
            'llm_test_case_capability': SimpleNamespace(input='structured')}
    evaluator = runner.CaseEvaluator(['reverse_validation', 'dataqa_capability'], Config())
    asyncio.run(evaluator.evaluate_case(case))
    assert calls == (['primary', 'model2', 'model3'] if expected_review else ['primary'])
    assert case['is_success'] is structured_pass


def test_run_async_false_and_throttle_apply_to_reviews_too(monkeypatch):
    starts = []

    async def hook(query, label):
        starts.append(asyncio.get_running_loop().time())
        return label != 'primary'

    install(monkeypatch, hook)

    async def scenario():
        conf = Config(**{'evluate.run_async': False, 'evluate.throttle_value': 0.01})
        evaluator = runner.CaseEvaluator(['reverse_validation'], conf)
        assert evaluator.concurrency == 1
        await evaluator.evaluate_case(row('q'))

    asyncio.run(scenario())
    assert len(starts) == 3
    assert all(b - a >= 0.009 for a, b in zip(starts, starts[1:]))


@pytest.mark.parametrize('values', [{'evluate.max_concurrent': 0},
                                  {'evluate.metric_timeout': 0},
                                  {'evluate.throttle_value': -1},
                                  {'retry.eval_max_retries': -1}])
def test_invalid_evaluation_config_fails_fast(values):
    with pytest.raises(ValueError):
        runner.CaseEvaluator(['reverse_validation'], Config(**values))
