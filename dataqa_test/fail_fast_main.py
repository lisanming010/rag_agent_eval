"""MySQL 用例 fail-fast 入口：遇到请求或上游 API 错误后停止派发。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataqa_test.authenticated_dataqa import create_refreshing_dataqa
from dataqa_test.main import (
    CASES_DIR,
    DEFAULT_OUTPUT_DIR,
    _make_run_id,
    filter_cases_by_data_domain,
    load_cases,
)
from dataqa_test.processor import _process_job, resolve_run_settings
from dataqa_test.result_writer import AsyncResultWriter, write_manifest

UPSTREAM_ERROR_CODE = "API_UPSTREAM_UNAVAILABLE"
REQUEST_FAILED_STATUS = "request_failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="执行 MySQL DataQA 用例，遇到请求或上游 API 错误立即停止派发"
    )
    parser.add_argument("--input", "-i", default=str(CASES_DIR))
    parser.add_argument("--output-dir", "-o", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--query-field")
    parser.add_argument("--data-domain", default="mysql")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--submit-delay", type=float, default=0)
    return parser.parse_args()


def upstream_error_details(result: dict[str, Any]) -> dict[str, Any] | None:
    """从尚未写盘的内部 raw 中查找上游 API 错误。"""
    case_id = result.get("case", {}).get("id")
    for turn in result.get("turns", []):
        for request in turn.get("_raw", {}).get("requests", []):
            response = request.get("response") or {}
            metadata = ((response.get("content") or {}).get("metadata") or {})
            if str(metadata.get("error_code")) != UPSTREAM_ERROR_CODE:
                continue
            return {
                "case_id": case_id,
                "turn_index": turn.get("index"),
                "query": turn.get("query"),
                "stage": request.get("stage"),
                "attempt": request.get("attempt"),
                "auth_attempt": request.get("auth_attempt"),
                "dataqa_http_status": request.get("status_code"),
                "endpoint": metadata.get("endpoint"),
                "upstream_http_status": metadata.get("http_status_code"),
                "business_code": metadata.get("business_code"),
                "query_status": metadata.get("query_status"),
                "error_code": metadata.get("error_code"),
                "message": metadata.get("message"),
            }
    return None


def failure_trigger_details(result: dict[str, Any]) -> dict[str, Any] | None:
    """上游 API 错误或问数请求失败均触发停止派发。"""
    upstream_error = upstream_error_details(result)
    if upstream_error is not None:
        return {"trigger_type": "upstream_api_error", **upstream_error}

    execution = result.get("execution", {})
    if str(execution.get("status")) != REQUEST_FAILED_STATUS:
        return None
    failed_turn = next(
        (
            turn
            for turn in result.get("turns", [])
            if str(turn.get("status")) == REQUEST_FAILED_STATUS
        ),
        {},
    )
    return {
        "trigger_type": "request_failed",
        "case_id": result.get("case", {}).get("id"),
        "turn_index": failed_turn.get("index"),
        "query": failed_turn.get("query"),
        "error": failed_turn.get("error") or execution.get("error"),
    }


def run_cases_fail_fast(
    rows: list[dict[str, Any]],
    *,
    query_field: str | None,
    settings: dict[str, int | float],
    on_result: Callable[[dict[str, Any]], None],
) -> tuple[dict[str, Any] | None, int]:
    """只维持 max_workers 个在途任务，触发后不再派发新用例。"""
    agent = create_refreshing_dataqa()
    max_workers = int(settings["max_workers"])
    submit_delay = float(settings["submit_delay"])
    rows_iterator = iter(rows)
    futures: dict[Future[dict[str, Any]], dict[str, Any]] = {}
    completed = 0
    trigger: dict[str, Any] | None = None

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        job = {"case": row, "result": None}
        _process_job(
            agent,
            job,
            query_field=query_field,
            max_retries=int(settings["max_retries"]),
            retry_delay=float(settings["retry_delay"]),
            on_result=None,
        )
        return job["result"]

    def submit_next(executor: ThreadPoolExecutor) -> bool:
        try:
            row = next(rows_iterator)
        except StopIteration:
            return False
        futures[executor.submit(execute, row)] = row
        if submit_delay > 0:
            time.sleep(submit_delay)
        return True

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(min(max_workers, len(rows))):
            submit_next(executor)

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                result = future.result()
                error = failure_trigger_details(result)
                on_result(result)
                completed += 1
                print(f"[{completed}/{len(rows)}] DataQA 调度完成", flush=True)
                if trigger is None and error is not None:
                    trigger = error
                    print(
                        "[FAIL-FAST] " + json.dumps(trigger, ensure_ascii=False),
                        flush=True,
                    )

            if trigger is None:
                while len(futures) < max_workers and submit_next(executor):
                    pass

    return trigger, completed


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    source_rows = load_cases(input_path)
    rows = filter_cases_by_data_domain(source_rows, args.data_domain)
    if not rows:
        print("没有匹配的数据域用例", file=sys.stderr)
        return 2

    run_id = _make_run_id(args.run_id)
    run_dir = Path(args.output_dir) / run_id
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(exist_ok=False)
    settings = resolve_run_settings(
        max_workers=args.max_workers,
        submit_delay=args.submit_delay,
    )
    started_at = datetime.now().astimezone()
    writer = AsyncResultWriter(
        run_dir,
        queue_maxsize=max(2, int(settings["max_workers"]) * 2),
    )
    writer.start()
    trigger: dict[str, Any] | None = None
    completed = 0
    try:
        trigger, completed = run_cases_fail_fast(
            rows,
            query_field=args.query_field,
            settings=settings,
            on_result=writer.submit,
        )
    finally:
        writer.close()

    completed_at = datetime.now().astimezone()
    stats = writer.summary
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": round((completed_at - started_at).total_seconds() * 1000),
        "input": {
            "path": str(input_path),
            "source_case_count": len(source_rows),
            "case_count": len(rows),
        },
        "settings": {
            **settings,
            "query_field": args.query_field,
            "data_domain": args.data_domain,
            "raw_recording": "all_requests",
            "write_mode": "async_single_writer",
            "dispatch_mode": "bounded_fail_fast",
        },
        "resume_supported": False,
        "aborted": trigger is not None,
        "abort_condition": [REQUEST_FAILED_STATUS, UPSTREAM_ERROR_CODE],
        "abort_trigger": trigger,
        "completed_before_stop": completed,
        "summary": stats,
        "artifacts": {"results": "results.jsonl", "raw_dir": "raw"},
    }
    manifest_path = write_manifest(run_dir, manifest)
    print(f"运行目录: {run_dir}", flush=True)
    print(f"清单: {manifest_path}", flush=True)
    print(json.dumps(stats, ensure_ascii=False), flush=True)
    if trigger is not None:
        print(
            "检测到 fail-fast 错误，已停止继续派发: "
            + json.dumps(trigger, ensure_ascii=False),
            file=sys.stderr,
            flush=True,
        )
        return 3
    return 0 if stats.get("success", 0) == stats["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
