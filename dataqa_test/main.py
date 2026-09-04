"""临时问数智能体测试入口。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

# 兼容 `python dataqa_test/main.py` 直接执行方式。
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataqa_test.processor import (
    resolve_run_settings,
    run_cases,
)
from dataqa_test.result_writer import (
    AsyncResultWriter,
    write_manifest,
)

CASES_DIR = Path(__file__).resolve().parent / "cases"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 JSON/JSONL 用例，批量调用 DataQA 并保存精简结果"
    )
    parser.add_argument(
        "-i",
        "--input",
        default=str(CASES_DIR),
        help="输入 JSON/JSONL 文件或目录；默认读取 dataqa_test/cases",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="运行结果根目录；每次执行会在其下创建 run_id 子目录",
    )
    parser.add_argument(
        "--run-id",
        help="指定本次运行 ID；默认使用当前时间生成",
    )
    parser.add_argument(
        "--query-field",
        help="问题字段；不传时依次识别 turns、query、示例提问、question",
    )
    parser.add_argument(
        "--data-domain",
        default="mysql",
        help="只执行 required_data_domains 包含该值的用例；默认 mysql，传 all 不过滤",
    )
    parser.add_argument("--max-workers", type=int, help="并发数，默认读取 config.yaml")
    parser.add_argument(
        "--submit-delay",
        type=float,
        help="任务提交间隔（秒），默认读取 config.yaml",
    )
    return parser.parse_args()


def _read_json_cases(json_path: Path) -> list[dict]:
    """读取单个 JSON 用例文件，支持顶层数组或 {"cases": [...]}。"""
    with json_path.open("r", encoding="utf-8-sig") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        payload = payload.get("cases")
    if not isinstance(payload, list):
        raise ValueError(
            f"JSON 用例格式错误: {json_path}；顶层必须是数组或包含 cases 数组"
        )
    if not all(isinstance(case, dict) for case in payload):
        raise ValueError(f"JSON 用例格式错误: {json_path}；每条用例必须是对象")
    return payload


def _read_jsonl_cases(jsonl_path: Path) -> list[dict]:
    """读取每行一个 JSON 对象的 JSONL 用例文件。"""
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL 格式错误: {jsonl_path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"JSONL 格式错误: {jsonl_path}:{line_number}；每行必须是对象"
                )
            rows.append(row)
    return rows


def _read_case_file(case_path: Path) -> list[dict]:
    suffix = case_path.suffix.lower()
    if suffix == ".json":
        rows = _read_json_cases(case_path)
    elif suffix == ".jsonl":
        rows = _read_jsonl_cases(case_path)
    else:
        raise ValueError(f"输入文件必须为 JSON 或 JSONL: {case_path}")
    for row in rows:
        row.setdefault("dataset", case_path.stem)
    return rows


def load_cases(input_path: Path) -> list[dict]:
    """从 JSON/JSONL 文件或目录读取用例，目录内文件按名称排序合并。"""
    if input_path.is_file():
        return _read_case_file(input_path)
    if not input_path.is_dir():
        raise FileNotFoundError(f"输入路径不存在: {input_path}")

    case_files = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".json", ".jsonl"}
    )
    if not case_files:
        raise ValueError(f"目录中没有 JSON/JSONL 用例文件: {input_path}")

    rows: list[dict] = []
    for case_path in case_files:
        rows.extend(_read_case_file(case_path))
    return rows


def filter_cases_by_data_domain(
    rows: list[dict], data_domain: str
) -> list[dict]:
    """按 required_data_domains 做大小写不敏感的精确匹配。"""
    target = data_domain.strip().casefold()
    if not target:
        raise ValueError("data_domain 不能为空")
    if target == "all":
        return rows

    selected = []
    for row in rows:
        domains = row.get("required_data_domains")
        if isinstance(domains, str):
            normalized = {domains.strip().casefold()}
        elif isinstance(domains, list):
            normalized = {
                str(item).strip().casefold()
                for item in domains
                if str(item).strip()
            }
        else:
            normalized = set()
        if target in normalized:
            selected.append(row)
    return selected


def _make_run_id(value: str | None) -> str:
    if value:
        if not re.fullmatch(r"[0-9A-Za-z_.-]+", value) or value in {".", ".."}:
            raise ValueError("run_id 只能包含字母、数字、下划线、点和连字符")
        return value
    return datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    run_id = _make_run_id(args.run_id)
    run_dir = Path(args.output_dir) / run_id
    source_rows = load_cases(input_path)
    rows = filter_cases_by_data_domain(source_rows, args.data_domain)
    if not rows:
        print(
            f"没有匹配 data_domain={args.data_domain!r} 的测试用例: {input_path}",
            file=sys.stderr,
        )
        return 2

    settings = resolve_run_settings(
        max_workers=args.max_workers,
        submit_delay=args.submit_delay,
    )
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir(exist_ok=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"运行目录已存在，请更换 run_id: {run_dir}"
        ) from exc

    started_at = datetime.now().astimezone()
    queue_maxsize = max(2, int(settings["max_workers"]) * 2)
    writer = AsyncResultWriter(run_dir, queue_maxsize=queue_maxsize)
    writer.start()
    try:
        run_cases(
            rows,
            query_field=args.query_field,
            settings=settings,
            on_result=writer.submit,
        )
    finally:
        writer.close()
    result_path = writer.result_path

    completed_at = datetime.now().astimezone()
    stats = writer.summary
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": round(
            (completed_at - started_at).total_seconds() * 1000
        ),
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
            "writer_queue_maxsize": queue_maxsize,
        },
        "resume_supported": False,
        "summary": stats,
        "artifacts": {
            "results": result_path.relative_to(run_dir).as_posix(),
            "raw_dir": "raw" if (run_dir / "raw").exists() else None,
        },
    }
    manifest_path = write_manifest(run_dir, manifest)

    print(f"测试完成，运行目录: {run_dir}")
    print(f"结果: {result_path}")
    print(f"清单: {manifest_path}")
    print(json.dumps(stats, ensure_ascii=False))
    return 0 if stats.get("success", 0) == stats["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
