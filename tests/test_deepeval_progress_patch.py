"""Exercise DeepEval's real async metric dispatcher without LLM/network calls."""

import asyncio
from io import StringIO

import pytest
from deepeval.errors import MissingTestCaseParamsError
from deepeval.metrics import indicator
from rich.console import Console
from rich.progress import Progress

from evaluator.deepeval_patch import patch_evaluation_progress


@pytest.fixture(autouse=True)
def install_progress_patch(monkeypatch):
    original = indicator.safe_a_measure
    while getattr(original, "_rga_progress_cleanup_patched", False):
        original = original.__wrapped__
    monkeypatch.setattr(indicator, "safe_a_measure", original)
    patch_evaluation_progress()


class FakeMetric:
    def __init__(self, error=None, started=None, release=None, passed=True):
        self.exception = error
        self.started = started
        self.release = release
        self.passed = passed
        self.error = None
        self.success = None
        self.skipped = False
        self.score = None

    async def a_measure(self, test_case, **kwargs):
        if self.started is not None:
            self.started.set()
        if self.release is not None:
            await self.release.wait()
        if self.exception is not None:
            raise self.exception
        self.score = 1.0 if self.passed else 0.0
        self.success = self.passed


def make_progress(total=1):
    progress = Progress(console=Console(file=StringIO()), auto_refresh=False)
    task_id = progress.add_task("case", total=total)
    return progress, task_id, progress.tasks[0]


async def dispatch(metrics, progress, task_id, ignore_errors=True, skip=False):
    # This is the same branch used by evaluate()'s per-case progress bars.
    await indicator.measure_metrics_with_indicator(
        metrics=metrics,
        test_case=object(),
        cached_test_case=None,
        ignore_errors=ignore_errors,
        skip_on_missing_params=skip,
        show_indicator=False,
        progress=progress,
        pbar_eval_id=task_id,
    )


@pytest.mark.parametrize("error", [RuntimeError("HTTP 403"), TimeoutError("timeout")])
def test_ignored_error_does_not_leave_a_progress_row(error):
    metric = FakeMetric(error)
    progress, task_id, task = make_progress()

    asyncio.run(dispatch([metric], progress, task_id))

    assert metric.error == str(error)
    assert metric.success is False
    assert not progress.tasks
    assert task.completed == 1


@pytest.mark.parametrize("skip", [False, True])
def test_missing_parameters_complete_progress_when_ignored_or_skipped(skip):
    metric = FakeMetric(MissingTestCaseParamsError("missing expected_output"))
    progress, task_id, task = make_progress()

    asyncio.run(dispatch([metric], progress, task_id, skip=skip))

    assert metric.skipped is skip
    assert not progress.tasks
    assert task.completed == 1


@pytest.mark.parametrize("passed", [False, True])
def test_normal_result_advances_exactly_once(passed):
    metric = FakeMetric(passed=passed)
    progress, task_id, task = make_progress()

    asyncio.run(dispatch([metric], progress, task_id))

    assert metric.success is passed
    assert metric.error is None
    assert not progress.tasks
    assert task.completed == 1


def test_concurrent_metrics_do_not_remove_progress_before_last_metric_finishes():
    async def scenario():
        fast_done, slow_started, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
        metrics = [
            FakeMetric(),
            FakeMetric(RuntimeError("HTTP 403"), started=fast_done),
            FakeMetric(started=slow_started, release=release),
        ]
        progress, task_id, task = make_progress(total=3)
        pending = asyncio.create_task(dispatch(metrics, progress, task_id))
        try:
            await asyncio.wait_for(fast_done.wait(), timeout=2)
            await asyncio.wait_for(slow_started.wait(), timeout=2)
            assert task.completed == 2
            assert len(progress.tasks) == 1
        finally:
            release.set()
            await asyncio.wait_for(pending, timeout=2)
        assert not progress.tasks
        assert task.completed == 3

    asyncio.run(scenario())


@pytest.mark.parametrize("ignore_errors", [False, True])
def test_cancellation_cleans_progress_and_preserves_propagation(ignore_errors):
    async def scenario():
        started, release = asyncio.Event(), asyncio.Event()
        metric = FakeMetric(started=started, release=release)
        progress, task_id, task = make_progress()
        pending = asyncio.create_task(indicator.safe_a_measure(
            metric, object(), ignore_errors, False,
            progress=progress, pbar_eval_id=task_id,
        ))
        await asyncio.wait_for(started.wait(), timeout=2)
        pending.cancel()
        if ignore_errors:
            await pending
        else:
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert metric.success is False
        assert metric.error
        assert not progress.tasks
        assert task.completed == 1

    asyncio.run(scenario())


def test_non_ignored_error_is_still_raised_after_progress_cleanup():
    error = RuntimeError("HTTP 403")
    metric = FakeMetric(error)
    progress, task_id, task = make_progress()

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(dispatch([metric], progress, task_id, ignore_errors=False))

    assert caught.value is error
    assert not progress.tasks
    assert task.completed == 1


def test_patch_installation_is_idempotent():
    installed = indicator.safe_a_measure
    patch_evaluation_progress()
    patch_evaluation_progress()
    assert indicator.safe_a_measure is installed


@pytest.mark.parametrize("error", [None, RuntimeError("HTTP 403")])
def test_disabled_progress_does_not_change_metric_result(error):
    metric = FakeMetric(error)
    asyncio.run(dispatch([metric], None, None))
    assert metric.success is (error is None)
    assert metric.error == (str(error) if error else None)


def test_already_removed_progress_task_is_safe():
    metric = FakeMetric(RuntimeError("HTTP 403"))
    progress, task_id, task = make_progress()
    progress.remove_task(task_id)
    asyncio.run(dispatch([metric], progress, task_id))
    assert metric.error == "HTTP 403"
    assert not progress.tasks


def test_old_metric_signature_fallback_completes_progress():
    class OldMetric:
        async def a_measure(self, test_case):
            self.success = True

    metric = OldMetric()
    progress, task_id, task = make_progress()
    asyncio.run(dispatch([metric], progress, task_id))
    assert metric.success is True
    assert not progress.tasks
    assert task.completed == 1


def test_cleanup_failure_does_not_mask_original_exception(monkeypatch):
    original_update = indicator.update_pbar

    def broken_update(progress, task_id):
        if progress is not None:
            raise RuntimeError("terminal unavailable")
        return original_update(progress, task_id)

    monkeypatch.setattr(indicator, "update_pbar", broken_update)
    error = ValueError("original metric error")
    progress, task_id, _ = make_progress()
    with pytest.raises(ValueError) as caught:
        asyncio.run(dispatch([FakeMetric(error)], progress, task_id, ignore_errors=False))
    assert caught.value is error
