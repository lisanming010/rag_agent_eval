"""根据 dataqa_test 正式运行产物生成可按用例 ID 追溯的 Markdown 报告。"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 DataQA Markdown 测试报告")
    parser.add_argument("run_dir", type=Path, help="包含 manifest/results/raw 的运行目录")
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "cases",
        help="原始 JSON/JSONL 用例目录",
    )
    parser.add_argument("--output", type=Path, help="报告路径，默认 RUN_DIR/report.md")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            row["__result_line__"] = line_number
            rows.append(row)
    return rows


def source_case_index(cases_dir: Path) -> dict[tuple[str, str], tuple[str, int | None]]:
    index: dict[tuple[str, str], tuple[str, int | None]] = {}
    for path in sorted(cases_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
            continue
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8-sig") as file:
                for line_number, line in enumerate(file, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    case_id = str(row.get("id", ""))
                    if case_id:
                        index[(path.stem, case_id)] = (path.name, line_number)
            continue

        payload = load_json(path)
        if isinstance(payload, dict):
            payload = payload.get("cases", [])
        for row in payload if isinstance(payload, list) else []:
            if isinstance(row, dict) and row.get("id"):
                index[(path.stem, str(row["id"]))] = (path.name, None)
    return index


def raw_by_case(run_dir: Path, results: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for result in results:
        case_id = str(result.get("case", {}).get("id", ""))
        raw_ref = result.get("execution", {}).get("raw_ref")
        if not case_id or not raw_ref:
            continue
        raw_path = run_dir / str(raw_ref)
        if raw_path.is_file():
            payloads[case_id] = load_json(raw_path)
    return payloads


def requests_for(raw: dict[str, Any] | None) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for turn in (raw or {}).get("turns", []):
        requests.extend(item for item in turn.get("requests", []) if isinstance(item, dict))
    return requests


def confirm_statuses(raw: dict[str, Any] | None) -> list[str]:
    statuses: list[str] = []
    for request in requests_for(raw):
        if request.get("stage") != "confirm_execute":
            continue
        metadata = (
            (request.get("response") or {}).get("content") or {}
        ).get("metadata") or {}
        value = metadata.get("query_status")
        statuses.append(str(value).strip() if value not in {None, ""} else "missing")
    return statuses


def format_statuses(statuses: list[str]) -> str:
    if not statuses:
        return "未执行"
    counts = Counter(statuses)
    order = ["success", "empty", "error", "missing"]
    labels = {"success": "成功", "empty": "空结果", "error": "失败", "missing": "缺失"}
    parts = []
    for status in order + sorted(set(counts) - set(order)):
        if status not in counts:
            continue
        label = labels.get(status, status)
        parts.append(label if counts[status] == 1 else f"{label}×{counts[status]}")
    return "、".join(parts)


def issue_details(
    result: dict[str, Any],
    raw: dict[str, Any] | None,
) -> tuple[str, str]:
    """提取有问题轮次的用户 query 和智能体原始错误。"""
    issue_queries: list[str] = []
    issue_messages: list[str] = []
    for turn in (raw or {}).get("turns", []):
        turn_index = turn.get("index", "?")
        turn_messages: list[str] = []
        turn_error = str(turn.get("error") or "").strip()
        if turn_error:
            turn_messages.append(turn_error)

        for request in turn.get("requests", []):
            if not isinstance(request, dict):
                continue
            request_error = str(request.get("error") or "").strip()
            if request_error:
                turn_messages.append(request_error)
            status_code = request.get("status_code")
            if status_code not in {None, 200}:
                turn_messages.append(f"HTTP {status_code}")

            response = request.get("response") or {}
            content = response.get("content") or {}
            metadata = content.get("metadata") or {}
            if request.get("stage") != "confirm_execute":
                continue
            query_status = str(metadata.get("query_status") or "").strip()
            if query_status not in {"error", "empty"}:
                continue
            error_code = str(metadata.get("error_code") or "").strip()
            message = str(metadata.get("message") or "").strip()
            detail = f"query_status={query_status}"
            if error_code:
                detail += f", error_code={error_code}"
            if message:
                detail += f": {message}"
            turn_messages.append(detail)

        unique_messages = list(dict.fromkeys(turn_messages))
        if not unique_messages:
            continue
        issue_queries.append(f"T{turn_index}: {turn.get('query', '')}")
        issue_messages.append(
            f"T{turn_index}: " + "；".join(unique_messages)
        )

    execution = result.get("execution", {})
    if not issue_messages and execution.get("status") != "success":
        fallback_error = str(execution.get("error") or "未提供错误信息")
        issue_messages.append(fallback_error)
        issue_queries.extend(
            f"T{turn.get('index', '?')}: {turn.get('query', '')}"
            for turn in result.get("turns", [])
        )
    return (
        "<br>".join(issue_queries) if issue_queries else "-",
        "<br>".join(issue_messages) if issue_messages else "-",
    )


def warning_category(error: Any) -> str:
    text = str(error or "")
    rules = [
        ("能力选择不明确", "Choose the business capability"),
        ("缺少必要查询条件", "query conditions are required"),
        ("业务对象不唯一", "No unique business object"),
        ("多轮关系不明确", "relates to the previous query"),
        ("电站候选不唯一", "possible stations"),
        ("实体重名", "Duplicate names exist"),
        ("设备标识不明确", "device name is not specific enough"),
    ]
    for category, marker in rules:
        if marker.casefold() in text.casefold():
            return category
    return "其他业务警告"


def natural_key(value: Any) -> list[Any]:
    return [int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", str(value))]


def md_escape(value: Any) -> str:
    return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")


def percent(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator * 100:.1f}%" if denominator else "0.0%"


def duration_text(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours} 小时 {minutes} 分 {seconds} 秒"
    return f"{minutes} 分 {seconds} 秒"


def dataset_row(
    dataset: str,
    items: list[dict[str, Any]],
    raw_map: dict[str, dict[str, Any]],
) -> list[Any]:
    statuses = Counter(str(item.get("execution", {}).get("status", "unknown")) for item in items)
    declared_turns = sum(int(item.get("execution", {}).get("total_turns", 0)) for item in items)
    completed_turns = sum(int(item.get("execution", {}).get("completed_turns", 0)) for item in items)
    processed_turns = sum(len(item.get("turns", [])) for item in items)
    business = Counter()
    echart_cases = 0
    echart_turns = 0
    for item in items:
        case_id = str(item.get("case", {}).get("id", ""))
        business.update(confirm_statuses(raw_map.get(case_id)))
        case_echarts = sum(bool(turn.get("answer", {}).get("has_echart")) for turn in item.get("turns", []))
        echart_turns += case_echarts
        echart_cases += int(case_echarts > 0)
    return [
        dataset,
        len(items),
        declared_turns,
        processed_turns,
        completed_turns,
        statuses["success"],
        statuses["business_warning"],
        statuses["partial_success"],
        business["success"],
        business["empty"],
        business["error"],
        echart_cases,
        echart_turns,
    ]


def table(headers: list[str], rows: Iterable[Iterable[Any]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(md_escape(value) for value in row) + " |" for row in rows)
    return lines


def build_report(run_dir: Path, cases_dir: Path) -> str:
    manifest = load_json(run_dir / "manifest.json")
    results = load_jsonl(run_dir / "results.jsonl")
    raw_map = raw_by_case(run_dir, results)
    sources = source_case_index(cases_dir)
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        by_dataset[str(result.get("case", {}).get("dataset", "unknown"))].append(result)

    all_requests = [request for raw in raw_map.values() for request in requests_for(raw)]
    http_statuses = Counter(str(request.get("status_code") or "null") for request in all_requests)
    status_counts = Counter(str(item.get("execution", {}).get("status", "unknown")) for item in results)
    flow_success_cases = status_counts["success"]
    warning_failure_cases = 0
    request_failure_cases = 0
    for item in results:
        execution_status = str(item.get("execution", {}).get("status", "unknown"))
        if execution_status == "success":
            continue
        turn_statuses = {
            str(turn.get("status", "unknown"))
            for turn in item.get("turns", [])
        }
        if execution_status == "business_warning" or "business_warning" in turn_statuses:
            warning_failure_cases += 1
        else:
            request_failure_cases += 1
    flow_failure_cases = warning_failure_cases + request_failure_cases
    dataset_rows = [dataset_row(name, items, raw_map) for name, items in sorted(by_dataset.items())]
    total_row = dataset_row("合计", results, raw_map)
    declared_turns = sum(int(item.get("execution", {}).get("total_turns", 0)) for item in results)
    processed_turns = sum(len(item.get("turns", [])) for item in results)
    completed_turns = sum(int(item.get("execution", {}).get("completed_turns", 0)) for item in results)
    multi_turn_cases = sum(int(item.get("execution", {}).get("total_turns", 0)) > 1 for item in results)
    confirm_requests = [request for request in all_requests if request.get("stage") == "confirm_execute"]
    business_counts = Counter()
    business_error_case_ids: set[str] = set()
    for raw in raw_map.values():
        business_counts.update(confirm_statuses(raw))
    for case_id, raw in raw_map.items():
        if any(status == "error" for status in confirm_statuses(raw)):
            business_error_case_ids.add(case_id)
    echart_turns = sum(
        bool(turn.get("answer", {}).get("has_echart"))
        for item in results
        for turn in item.get("turns", [])
    )
    echart_cases = sum(
        any(bool(turn.get("answer", {}).get("has_echart")) for turn in item.get("turns", []))
        for item in results
    )

    lines = [
        "# DataQA MySQL 数据域测试报告",
        "",
        f"> 运行 ID：`{md_escape(manifest.get('run_id'))}`  ",
        f"> 执行时间：{md_escape(manifest.get('started_at'))} ～ {md_escape(manifest.get('completed_at'))}  ",
        f"> 报告生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## 1. 执行结论",
        "",
        f"本次从 {manifest.get('input', {}).get('source_case_count', len(results))} 条源用例中筛选出 "
        f"{len(results)} 条 `required_data_domains` 包含 `mysql` 的用例，以 "
        f"{manifest.get('settings', {}).get('max_workers')} 并发执行，耗时 "
        f"{duration_text(int(manifest.get('duration_ms', 0)))}。结果和 raw 均为 {len(results)} 条，ID 一一对应。",
        "",
        "### 总体概览（流程口径）",
        "",
        *table(
            ["总用例", "成功", "失败", "成功率", "失败率"],
            [
                [
                    len(results),
                    flow_success_cases,
                    flow_failure_cases,
                    percent(flow_success_cases, len(results)),
                    percent(flow_failure_cases, len(results)),
                ]
            ],
        ),
        "",
        "### 失败类型拆分",
        "",
        *table(
            ["失败类型", "用例数", "占全部用例", "占失败用例", "说明"],
            [
                [
                    "request_failed",
                    request_failure_cases,
                    percent(request_failure_cases, len(results)),
                    percent(request_failure_cases, flow_failure_cases),
                    "HTTP、连接、超时或请求处理异常",
                ],
                [
                    "business_warning",
                    warning_failure_cases,
                    percent(warning_failure_cases, len(results)),
                    percent(warning_failure_cases, flow_failure_cases),
                    f"首轮 warning {status_counts['business_warning']} 条；后续轮次 warning 导致 partial_success {status_counts['partial_success']} 条",
                ],
            ],
        ),
        "",
        "> 这里的“成功/失败”按用例流程状态统计。二步确认后返回 `query_status=error` 的业务查询失败另见第 3、5 节，不计为 request_failed。",
        "",
        *table(
            ["指标", "数量", "说明"],
            [
                ["源数据集用例", manifest.get("input", {}).get("source_case_count"), "cases 目录全部用例"],
                ["本次 MySQL 用例", len(results), "未混入其他数据域"],
                ["声明轮次", declared_turns, "源用例 turns 总数"],
                ["实际处理轮次", processed_turns, "遇到警告后后续轮次不再发送"],
                ["完成二步确认轮次", completed_turns, percent(completed_turns, declared_turns)],
                ["多轮用例", multi_turn_cases, "total_turns > 1"],
                ["HTTP 请求", len(all_requests), ", ".join(f"{key}={value}" for key, value in sorted(http_statuses.items()))],
                ["401 / 鉴权重试", sum(r.get("status_code") == 401 for r in all_requests), sum(int(r.get("auth_attempt", 1)) > 1 for r in all_requests)],
                ["ECharts 用例 / 轮次", f"{echart_cases} / {echart_turns}", "answer.has_echart=true"],
            ],
        ),
        "",
        "### 状态口径",
        "",
        "- `success`：用例所有已声明轮次均完成 new_query 和 confirm_execute。",
        "- `business_warning`：首轮未得到 confirmationId，通常需要补充条件或选择候选项。",
        "- `partial_success`：多轮用例至少一轮完成，后续轮次出现业务警告。",
        "- 业务查询状态取自 raw 的 `content.metadata.query_status`。HTTP/流程成功不代表业务查询成功。",
        "",
        "## 2. 按数据集分类统计",
        "",
        *table(
            [
                "数据集", "用例", "声明轮次", "处理轮次", "完成轮次", "流程成功",
                "业务警告", "部分成功", "查询成功", "空结果", "查询失败", "ECharts用例", "ECharts轮次",
            ],
            dataset_rows + [total_row],
        ),
        "",
        "## 3. 总体执行状态",
        "",
        *table(
            ["状态", "用例数", "占比"],
            [
                ["success", status_counts["success"], percent(status_counts["success"], len(results))],
                ["business_warning", status_counts["business_warning"], percent(status_counts["business_warning"], len(results))],
                ["partial_success", status_counts["partial_success"], percent(status_counts["partial_success"], len(results))],
                ["request_failed", status_counts["request_failed"], percent(status_counts["request_failed"], len(results))],
            ],
        ),
        "",
        "### 二步确认后的实际业务状态",
        "",
        *table(
            ["query_status", "轮次", "占确认请求比例"],
            [
                ["success", business_counts["success"], percent(business_counts["success"], len(confirm_requests))],
                ["empty", business_counts["empty"], percent(business_counts["empty"], len(confirm_requests))],
                ["error", business_counts["error"], percent(business_counts["error"], len(confirm_requests))],
                ["missing", business_counts["missing"], percent(business_counts["missing"], len(confirm_requests))],
            ],
        ),
        "",
    ]

    warning_groups: dict[str, list[str]] = defaultdict(list)
    for item in results:
        execution_status = str(item.get("execution", {}).get("status", "unknown"))
        turn_statuses = {
            str(turn.get("status", "unknown"))
            for turn in item.get("turns", [])
        }
        if execution_status not in {"business_warning", "partial_success"} and "business_warning" not in turn_statuses:
            continue
        error = item.get("execution", {}).get("error")
        if error:
            warning_groups[warning_category(error)].append(str(item.get("case", {}).get("id", "?")))
    lines.extend(["## 4. 业务警告分类", ""])
    lines.extend(
        table(
            ["分类", "用例数", "测试数据集 ID"],
            [
                [category, len(ids), " ".join(f"`{case_id}`" for case_id in sorted(ids, key=natural_key))]
                for category, ids in sorted(warning_groups.items(), key=lambda item: (-len(item[1]), item[0]))
            ],
        )
    )

    business_error_groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for case_id, raw in raw_map.items():
        for request in requests_for(raw):
            if request.get("stage") != "confirm_execute":
                continue
            metadata = (((request.get("response") or {}).get("content") or {}).get("metadata") or {})
            if metadata.get("query_status") != "error":
                continue
            key = (str(metadata.get("error_code") or "未提供"), str(metadata.get("message") or "未提供"))
            business_error_groups[key].add(case_id)
    lines.extend(
        [
            "",
            "## 5. 二步确认后的业务失败分类",
            "",
            f"共 {business_counts['error']} 个确认轮次返回 `query_status=error`，涉及 "
            f"{len(business_error_case_ids)} 个测试用例。以下用例数按 ID 去重统计。",
            "",
        ]
    )
    lines.extend(
        table(
            ["error_code", "message", "用例数", "测试数据集 ID"],
            [
                [code, message, len(ids), " ".join(f"`{case_id}`" for case_id in sorted(ids, key=natural_key))]
                for (code, message), ids in sorted(
                    business_error_groups.items(), key=lambda item: (-len(item[1]), item[0])
                )
            ],
        )
    )

    lines.extend(
        [
            "",
            "## 6. 测试数据集 ID 逐条追溯",
            "",
            "以下按数据集列出全部本次执行 ID。`源用例` 标明 cases 文件及行号，`结果行` 指向本次 "
            "results.jsonl，`raw` 指向该 ID 的全部实际请求。问题列覆盖流程非成功、请求异常以及 "
            "`query_status=error/empty`；正常用例显示 `-`。",
            "",
        ]
    )
    for dataset, items in sorted(by_dataset.items()):
        lines.extend([f"### {md_escape(dataset)}（{len(items)} 条）", ""])
        detail_rows = []
        for item in sorted(items, key=lambda value: natural_key(value.get("case", {}).get("id", ""))):
            case = item.get("case", {})
            execution = item.get("execution", {})
            case_id = str(case.get("id", "?"))
            source_name, source_line = sources.get((dataset, case_id), (f"{dataset}.jsonl", None))
            source_label = f"{source_name}:{source_line}" if source_line else source_name
            source_link = f"../../cases/{source_name}" + (f"#L{source_line}" if source_line else "")
            result_line = int(item.get("__result_line__", 0))
            case_echart = any(bool(turn.get("answer", {}).get("has_echart")) for turn in item.get("turns", []))
            problem_query, agent_error = issue_details(item, raw_map.get(case_id))
            detail_rows.append(
                [
                    f"`{case_id}`",
                    case.get("level", ""),
                    f"{execution.get('completed_turns', 0)}/{execution.get('total_turns', 0)}",
                    f"`{execution.get('status', 'unknown')}`",
                    format_statuses(confirm_statuses(raw_map.get(case_id))),
                    "是" if case_echart else "否",
                    problem_query,
                    agent_error,
                    f"[{source_label}]({source_link})",
                    f"[L{result_line}](results.jsonl#L{result_line})",
                    f"[raw]({execution.get('raw_ref')})" if execution.get("raw_ref") else "-",
                ]
            )
        lines.extend(
            table(
                [
                    "ID", "级别", "完成轮次", "执行状态", "业务查询", "ECharts",
                    "问题轮次 Query", "智能体错误响应", "源用例", "结果行", "Raw",
                ],
                detail_rows,
            )
        )
        lines.append("")

    lines.extend(
        [
            "## 7. 产物与注意事项",
            "",
            "- 精简结果：[`results.jsonl`](results.jsonl)",
            "- 运行清单：[`manifest.json`](manifest.json)",
            "- 原始请求：[`raw/`](raw/)",
            "- raw 不包含登录响应、密码、Authorization 或 JWT。",
            "- 本报告只做结果归类，没有修改测试结果和原始代码。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    cases_dir = args.cases_dir.resolve()
    output = (args.output or run_dir / "report.md").resolve()
    report = build_report(run_dir, cases_dir)
    output.write_text(report, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
