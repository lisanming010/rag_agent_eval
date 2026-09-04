"""DataQA 临时测试的单线程异步结果写入器。"""

from __future__ import annotations

import json
import queue
import re
import threading
from pathlib import Path
from typing import Any

_STOP = object()


def _safe_name(value: Any) -> str:
    name = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "unknown"))
    return name.strip("._") or "unknown"


def _write_json_atomic(path: Path, payload: Any) -> None:
    """先写同目录临时文件，再原子替换正式文件。"""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(path)


def _prepare_result(
    result: dict[str, Any],
    *,
    run_dir: Path,
    used_names: set[str],
    fallback_index: int,
) -> dict[str, Any]:
    """写出单条用例 raw，并清理精简结果中的内部字段。"""
    case = result.get("case", {})
    case_id = case.get("id") or f"case_{fallback_index}"
    filename = f"{_safe_name(case_id)}.json"
    if filename in used_names:
        raise ValueError(f"用例 ID 生成了重复的 raw 文件名: {case_id}")
    used_names.add(filename)

    raw_turns = []
    for turn in result.get("turns", []):
        raw = turn.pop("_raw", {})
        raw_turns.append(
            {
                "index": turn.get("index"),
                "query": turn.get("query"),
                "status": turn.get("status"),
                "error": turn.get("error"),
                "requests": raw.get("requests", []),
            }
        )

    raw_path = run_dir / "raw" / filename
    raw_payload = {
        "case": {
            "id": case_id,
            "dataset": case.get("dataset"),
            "module": case.get("module"),
        },
        "session_id": result.get("execution", {}).get("session_id"),
        "turns": raw_turns,
    }
    _write_json_atomic(raw_path, raw_payload)
    result.setdefault("execution", {})["raw_ref"] = (
        raw_path.relative_to(run_dir).as_posix()
    )
    return result


class AsyncResultWriter:
    """通过有界队列把用例结果交给单独线程逐条落盘。"""

    def __init__(self, run_dir: Path, queue_maxsize: int = 8):
        if queue_maxsize < 1:
            raise ValueError("queue_maxsize 必须大于等于 1")
        self.run_dir = run_dir
        self.result_path = run_dir / "results.jsonl"
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=queue_maxsize)
        self._thread = threading.Thread(
            target=self._run,
            name="dataqa-result-writer",
            daemon=False,
        )
        self._started = False
        self._closed = False
        self._error: BaseException | None = None
        self._summary: dict[str, int] = {"total": 0}

    def start(self) -> None:
        if self._started:
            raise RuntimeError("写入线程已经启动")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._started = True
        self._thread.start()

    def submit(self, result: dict[str, Any]) -> None:
        """提交一条结果；队列满时施加背压，避免 raw 无限堆积。"""
        if not self._started or self._closed:
            raise RuntimeError("写入线程尚未启动或已经关闭")
        while True:
            if self._error is not None:
                raise RuntimeError("写入线程已经失败") from self._error
            try:
                self._queue.put(result, timeout=0.2)
                return
            except queue.Full:
                continue

    def close(self) -> None:
        """等待队列内结果全部写完，并向调用方传播写入异常。"""
        if not self._started or self._closed:
            return
        self._closed = True
        while self._error is None:
            try:
                self._queue.put(_STOP, timeout=0.2)
                break
            except queue.Full:
                continue
        self._thread.join()
        if self._error is not None:
            raise RuntimeError("结果写入失败") from self._error

    @property
    def summary(self) -> dict[str, int]:
        if not self._closed:
            raise RuntimeError("写入线程关闭后才能读取完整汇总")
        return dict(self._summary)

    def _run(self) -> None:
        used_names: set[str] = set()
        written = 0
        try:
            with self.result_path.open("w", encoding="utf-8") as result_file:
                while True:
                    item = self._queue.get()
                    try:
                        if item is _STOP:
                            return
                        written += 1
                        result = _prepare_result(
                            item,
                            run_dir=self.run_dir,
                            used_names=used_names,
                            fallback_index=written,
                        )
                        result_file.write(
                            json.dumps(result, ensure_ascii=False) + "\n"
                        )
                        result_file.flush()
                        status = str(
                            result.get("execution", {}).get("status")
                            or "unknown"
                        )
                        self._summary["total"] += 1
                        self._summary[status] = (
                            self._summary.get(status, 0) + 1
                        )
                    finally:
                        self._queue.task_done()
        except BaseException as exc:
            self._error = exc


def prepare_results(
    results: list[dict[str, Any]],
    *,
    run_dir: Path,
) -> list[dict[str, Any]]:
    """离线或测试场景下串行准备一组结果。"""
    used_names: set[str] = set()
    for index, result in enumerate(results, start=1):
        _prepare_result(
            result,
            run_dir=run_dir,
            used_names=used_names,
            fallback_index=index,
        )
    return results


def write_results_jsonl(run_dir: Path, results: list[dict[str, Any]]) -> Path:
    """离线或测试场景下串行写入 JSONL。"""
    result_path = run_dir / "results.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False))
            file.write("\n")
    return result_path


def write_manifest(run_dir: Path, manifest: dict[str, Any]) -> Path:
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return manifest_path
