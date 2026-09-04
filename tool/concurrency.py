"""通用并发工具"""

import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from threading import Lock
from typing import Callable, NamedTuple


PROGRESS_INTERVAL_SECONDS = 1.0
HEARTBEAT_INTERVAL_SECONDS = 10.0


class ProgressEvent(NamedTuple):
    """线程池向外暴露的稳定进度事件。"""

    event: str
    task_name: str
    total: int = 0
    submitted: int = 0
    completed: int = 0
    succeeded: int = 0
    failed: int = 0
    running: int = 0
    waiting: int = 0
    elapsed_seconds: float = 0.0
    workers: int = 0
    submit_delay: float = 0.0
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]
_TERMINAL_LOCK = Lock()


def emit_terminal_progress(event: ProgressEvent) -> None:
    """默认终端适配器；独立于 logging 配置且保证多线程输出不交错。"""
    message = format_progress_event(event)
    with _TERMINAL_LOCK:
        print(message, flush=True)


def format_progress_event(event: ProgressEvent) -> str:
    """将语义进度事件格式化为单行终端文本。"""
    prefix = f"[PROGRESS] {event.task_name}"
    elapsed = _format_duration(event.elapsed_seconds)

    if event.event == "start":
        return (
            f"{prefix} 开始 | 总数={event.total} 并发数={event.workers} "
            f"提交间隔={event.submit_delay:g}s"
        )

    stats = (
        f"完成={event.completed}/{event.total} "
        f"({_percent(event.completed, event.total):.1f}%) "
        f"成功={event.succeeded} 失败={event.failed} "
        f"进行中={event.running} 待提交={event.waiting} 已耗时={elapsed}"
    )
    detail = f" | {_single_line(event.message)}" if event.message else ""

    labels = {
        "update": "进度",
        "heartbeat": "仍在运行",
        "retry": "重试",
        "failure": "失败",
        "finish": "完成",
    }
    label = labels.get(event.event, event.event)
    if event.event in {"retry", "failure"} and event.total == 0:
        return f"{prefix} {label}{detail}"
    return f"{prefix} {label} | {stats}{detail}"


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _percent(completed: int, total: int) -> float:
    return completed * 100 / total if total else 100.0


def _single_line(value: object) -> str:
    return " ".join(str(value).split())


def _describe_item(item: object) -> str:
    if isinstance(item, dict):
        for key in ("用例编号", "case_id", "id", "query"):
            value = item.get(key)
            if value is not None and str(value).strip():
                return _single_line(value)[:80]
    return _single_line(item)[:80]


def run_in_thread_pool(fn, items: list, max_workers: int = 8, submit_delay: float = 0,
                       task_name: str = "task",
                       on_progress: ProgressCallback | None = None):
    """
    通用线程池调度方法，对items列表中的每个元素并发执行fn

    :param fn: 可调用对象，接受单个item作为参数，可通过functools.partial绑定额外参数
    :param items: 待处理的数据列表
    :param max_workers: 最大线程数，默认8
    :param submit_delay: 提交间隔（秒），0 表示无间隔连续提交
    :param task_name: 任务名称，用于日志输出
    :param on_progress: 可选进度事件接收函数；默认即时输出到终端
    """
    total = len(items)
    completed = 0
    submitted = 0
    succeeded = 0
    failed = 0
    item_iter = iter(items)
    exhausted = False
    submit_delay = max(0, submit_delay)
    started_at = time.monotonic()
    next_submit_at = started_at
    next_heartbeat_at = started_at + HEARTBEAT_INTERVAL_SECONDS
    last_update_at = started_at
    progress_callback = on_progress or emit_terminal_progress

    def report_completed(done_futures, pending):
        nonlocal completed, succeeded, failed, last_update_at, next_heartbeat_at
        for future in done_futures:
            item = pending.pop(future)
            completed += 1
            try:
                future.result()
                succeeded += 1
            except Exception as e:
                failed += 1
                progress_callback(make_event(
                    "failure",
                    pending,
                    message=(
                        f"用例={_describe_item(item)} "
                        f"原因={type(e).__name__}: {_single_line(e)}"
                    ),
                ))

        now = time.monotonic()
        next_heartbeat_at = now + HEARTBEAT_INTERVAL_SECONDS
        if (
            completed == 1
            or completed == total
            or now - last_update_at >= PROGRESS_INTERVAL_SECONDS
        ):
            progress_callback(make_event("update", pending, now=now))
            last_update_at = now

    def make_event(event, pending, *, now=None, message=""):
        current_time = time.monotonic() if now is None else now
        return ProgressEvent(
            event=event,
            task_name=task_name,
            total=total,
            submitted=submitted,
            completed=completed,
            succeeded=succeeded,
            failed=failed,
            running=len(pending),
            waiting=max(0, total - submitted),
            elapsed_seconds=current_time - started_at,
            workers=max_workers,
            submit_delay=submit_delay,
            message=message,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {}
        progress_callback(make_event("start", pending, now=started_at))

        while pending or not exhausted:
            done_now = {future for future in pending if future.done()}
            if done_now:
                report_completed(done_now, pending)
                continue

            now = time.monotonic()
            if not exhausted and len(pending) < max_workers and now >= next_submit_at:
                try:
                    item = next(item_iter)
                except StopIteration:
                    exhausted = True
                else:
                    pending[executor.submit(fn, item)] = item
                    submitted += 1
                    next_submit_at = time.monotonic() + submit_delay
                    continue

            if not pending:
                if exhausted:
                    break
                time.sleep(max(0, next_submit_at - time.monotonic()))
                continue

            timeout_candidates = [
                max(0, next_heartbeat_at - time.monotonic())
            ]
            if not exhausted and len(pending) < max_workers:
                timeout_candidates.append(
                    max(0, next_submit_at - time.monotonic())
                )

            done_futures, _ = wait(
                pending,
                timeout=min(timeout_candidates),
                return_when=FIRST_COMPLETED,
            )
            if done_futures:
                report_completed(done_futures, pending)
                continue

            now = time.monotonic()
            if now >= next_heartbeat_at:
                progress_callback(make_event("heartbeat", pending, now=now))
                next_heartbeat_at = now + HEARTBEAT_INTERVAL_SECONDS

        progress_callback(make_event("finish", pending))
