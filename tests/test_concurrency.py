import importlib.util
import time
import unittest
from builtins import print as builtin_print
from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

CONCURRENCY_PATH = Path(__file__).resolve().parents[1] / "tool" / "concurrency.py"
CONCURRENCY_SPEC = importlib.util.spec_from_file_location("concurrency_under_test", CONCURRENCY_PATH)
concurrency = importlib.util.module_from_spec(CONCURRENCY_SPEC)
CONCURRENCY_SPEC.loader.exec_module(concurrency)


class ConcurrencyProgressTest(unittest.TestCase):
    def test_reports_completion_before_all_items_are_submitted(self):
        """完成进度必须在全量任务提交完之前可见。"""
        state = {"submitted": 0}
        events = []

        class TrackingExecutor:
            def __init__(self, *args, **kwargs):
                self._executor = RealThreadPoolExecutor(*args, **kwargs)

            def __enter__(self):
                self._executor.__enter__()
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return self._executor.__exit__(exc_type, exc_value, traceback)

            def submit(self, fn, *args, **kwargs):
                state["submitted"] += 1
                return self._executor.submit(fn, *args, **kwargs)

        with patch.object(concurrency, "ThreadPoolExecutor", TrackingExecutor):
            concurrency.run_in_thread_pool(
                lambda item: item,
                ["case-1", "case-2", "case-3"],
                max_workers=1,
                task_name="call_agent",
                on_progress=events.append,
            )

        updates = [event for event in events if event.event == "update"]
        self.assertTrue(updates)
        self.assertEqual(updates[0].submitted, 1)
        self.assertEqual(events[0].event, "start")
        self.assertEqual(events[-1].event, "finish")
        self.assertEqual(events[-1].succeeded, 3)
        self.assertEqual(events[-1].failed, 0)

    def test_reports_failure_and_includes_it_in_finish_summary(self):
        events = []

        def fail(_item):
            raise RuntimeError("request failed")

        concurrency.run_in_thread_pool(
            fail,
            ["case-1"],
            max_workers=1,
            task_name="call_agent",
            on_progress=events.append,
        )

        failures = [event for event in events if event.event == "failure"]
        self.assertEqual(len(failures), 1)
        self.assertIn("request failed", failures[0].message)
        self.assertEqual(events[-1].event, "finish")
        self.assertEqual(events[-1].completed, 1)
        self.assertEqual(events[-1].failed, 1)

    def test_emits_heartbeat_while_request_is_still_running(self):
        events = []

        with patch.object(concurrency, "HEARTBEAT_INTERVAL_SECONDS", 0.01):
            concurrency.run_in_thread_pool(
                lambda _item: time.sleep(0.04),
                ["case-1"],
                max_workers=1,
                task_name="call_agent",
                on_progress=events.append,
            )

        self.assertTrue(any(event.event == "heartbeat" for event in events))

    def test_terminal_adapter_flushes_each_event(self):
        event = concurrency.ProgressEvent(
            event="start",
            task_name="call_agent",
            total=3,
            workers=1,
            submit_delay=2,
        )

        with patch("builtins.print", wraps=builtin_print) as print_mock:
            concurrency.emit_terminal_progress(event)

        print_mock.assert_called_once_with(
            "[PROGRESS] call_agent 开始 | 总数=3 并发数=1 提交间隔=2s",
            flush=True,
        )
