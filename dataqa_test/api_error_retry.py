"""提取 API 错误用例，并在重跑完全干净后安全合并回基准运行。"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataqa_test.main import load_cases


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as file:
        return [json.loads(line) for line in file if line.strip()]


def iter_error_codes(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "error_code" and child:
                yield str(child)
            yield from iter_error_codes(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_error_codes(child)


def raw_payloads(run_dir: Path) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted((run_dir / "raw").glob("*.json")):
        payload = load_json(path)
        case_id = str((payload.get("case") or {}).get("id") or path.stem)
        payloads[case_id] = payload
    return payloads


def api_error_ids(run_dir: Path, error_code: str | None = None) -> set[str]:
    ids: set[str] = set()
    for case_id, payload in raw_payloads(run_dir).items():
        codes = set(iter_error_codes(payload))
        if error_code in codes if error_code else any(code.startswith("API_") for code in codes):
            ids.add(case_id)
    return ids


def command_extract(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    cases_dir = args.cases_dir.resolve()
    output = args.output.resolve()
    api_ids = api_error_ids(run_dir, args.error_code)
    request_failed_ids: set[str] = set()
    if args.include_request_failed:
        request_failed_ids = {
            str(row.get("case", {}).get("id"))
            for row in load_jsonl(run_dir / "results.jsonl")
            if row.get("execution", {}).get("status") == "request_failed"
        }
    selected_ids = api_ids | request_failed_ids
    source_cases = load_cases(cases_dir)
    selected = [case for case in source_cases if str(case.get("id")) in selected_ids]
    found_ids = {str(case.get("id")) for case in selected}
    missing = sorted(selected_ids - found_ids)
    if missing:
        raise ValueError(f"源 cases 中缺少 {len(missing)} 个 ID: {missing}")
    if len(found_ids) != len(selected):
        raise ValueError("源 cases 中存在重复 ID，无法生成无歧义重跑输入")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as file:
        for case in selected:
            file.write(json.dumps(case, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")

    datasets = Counter(str(case.get("dataset", "unknown")) for case in selected)
    print(
        json.dumps(
            {
                "base_run": str(run_dir),
                "error_code": args.error_code,
                "case_count": len(selected),
                "api_error_case_count": len(api_ids),
                "request_failed_case_count": len(request_failed_ids),
                "overlap_case_count": len(api_ids & request_failed_ids),
                "datasets": dict(sorted(datasets.items())),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def clean_retry_ids(run_dir: Path) -> tuple[set[str], set[str], set[str]]:
    results = load_jsonl(run_dir / "results.jsonl")
    all_ids = {str(row.get("case", {}).get("id")) for row in results}
    api_ids = api_error_ids(run_dir)
    non_success_ids = {
        str(row.get("case", {}).get("id"))
        for row in results
        if row.get("execution", {}).get("status") != "success"
    }
    return all_ids - api_ids - non_success_ids, api_ids, non_success_ids


def command_inspect(args: argparse.Namespace) -> int:
    clean_ids, api_ids, non_success_ids = clean_retry_ids(args.run_dir.resolve())
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "clean_success_count": len(clean_ids),
                "api_error_count": len(api_ids),
                "non_success_count": len(non_success_ids),
                "api_error_ids": sorted(api_ids),
                "non_success_ids": sorted(non_success_ids),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if api_ids or non_success_ids else 0


def atomic_write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def command_merge(args: argparse.Namespace) -> int:
    base_run = args.base_run.resolve()
    retry_run = args.retry_run.resolve()
    clean_ids, api_ids, non_success_ids = clean_retry_ids(retry_run)
    if api_ids or non_success_ids:
        raise RuntimeError(
            "重跑结果尚未完全干净，拒绝合并："
            f"API 错误 {len(api_ids)} 条，非 success {len(non_success_ids)} 条"
        )

    base_results = load_jsonl(base_run / "results.jsonl")
    retry_results = load_jsonl(retry_run / "results.jsonl")
    retry_by_id = {str(row["case"]["id"]): row for row in retry_results}
    base_ids = {str(row["case"]["id"]) for row in base_results}
    if clean_ids - base_ids:
        raise ValueError(f"重跑包含基准运行不存在的 ID: {sorted(clean_ids - base_ids)}")

    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    backup_dir = base_run / f"backup_before_api_retry_merge_{stamp}"
    backup_dir.mkdir(exist_ok=False)
    shutil.copy2(base_run / "results.jsonl", backup_dir / "results.jsonl")
    shutil.copy2(base_run / "manifest.json", backup_dir / "manifest.json")
    backup_raw = backup_dir / "raw"
    backup_raw.mkdir()
    for case_id in sorted(clean_ids):
        shutil.copy2(base_run / "raw" / f"{case_id}.json", backup_raw / f"{case_id}.json")

    merged = [retry_by_id.get(str(row["case"]["id"]), row) for row in base_results]
    results_tmp = base_run / "results.jsonl.tmp"
    with results_tmp.open("w", encoding="utf-8", newline="\n") as file:
        for row in merged:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    results_tmp.replace(base_run / "results.jsonl")
    for case_id in sorted(clean_ids):
        shutil.copy2(retry_run / "raw" / f"{case_id}.json", base_run / "raw" / f"{case_id}.json")

    manifest = load_json(base_run / "manifest.json")
    manifest["summary"] = dict(Counter(row.get("execution", {}).get("status", "unknown") for row in merged))
    manifest["summary"]["total"] = len(merged)
    manifest.setdefault("merge_history", []).append(
        {
            "merged_at": datetime.now().astimezone().isoformat(),
            "retry_run_id": load_json(retry_run / "manifest.json").get("run_id"),
            "replaced_case_count": len(clean_ids),
            "backup_dir": backup_dir.name,
        }
    )
    atomic_write_json(base_run / "manifest.json", manifest)
    print(json.dumps({"replaced": len(clean_ids), "backup": str(backup_dir)}, ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser("extract")
    extract.add_argument("run_dir", type=Path)
    extract.add_argument("cases_dir", type=Path)
    extract.add_argument("output", type=Path)
    extract.add_argument("--error-code", default="API_UPSTREAM_UNAVAILABLE")
    extract.add_argument("--include-request-failed", action="store_true")
    extract.set_defaults(handler=command_extract)

    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("run_dir", type=Path)
    inspect.set_defaults(handler=command_inspect)

    merge = subparsers.add_parser("merge")
    merge.add_argument("base_run", type=Path)
    merge.add_argument("retry_run", type=Path)
    merge.set_defaults(handler=command_merge)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
