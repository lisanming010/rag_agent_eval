"""DataQA 临时测试的用例执行与结果归一化。"""

from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable

from dataqa_test.authenticated_dataqa import create_refreshing_dataqa
from tool.concurrency import run_in_thread_pool
from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)

DEFAULT_QUERY_FIELDS = ("turns", "query", "示例提问", "question")
CASE_FIELDS = (
    "id",
    "dataset",
    "module",
    "level",
    "session_plan",
    "expected_modules",
    "required_data_domains",
)


def resolve_turns(
    row: dict[str, Any], query_field: str | None = None
) -> list[str]:
    """从用例中提取单轮或多轮问题。"""
    fields = (query_field,) if query_field else DEFAULT_QUERY_FIELDS
    for field in fields:
        value = row.get(field)
        if isinstance(value, list):
            turns = [str(item).strip() for item in value if str(item).strip()]
            if turns:
                return turns
        elif value is not None and str(value).strip():
            return [str(value).strip()]
    return []


def resolve_run_settings(
    *,
    max_workers: int | None = None,
    submit_delay: float | None = None,
) -> dict[str, int | float]:
    """合并 CLI 与项目配置，返回本次执行采用的实际参数。"""
    conf = ConfigReader.get_instance()
    settings: dict[str, int | float] = {
        "max_workers": (
            max_workers
            if max_workers is not None
            else conf.get("agents.http_agent.call_agent_th_max", 8)
        ),
        "submit_delay": (
            submit_delay
            if submit_delay is not None
            else conf.get("agents.http_agent.submit_delay", 0)
        ),
        "max_retries": conf.get(
            "agents.http_agent.call_agent_max_retries", 3
        ),
        "retry_delay": conf.get(
            "agents.http_agent.call_agent_retry_delay", 2
        ),
    }
    if settings["max_workers"] < 1:
        raise ValueError("max_workers 必须大于等于 1")
    if settings["submit_delay"] < 0:
        raise ValueError("submit_delay 不能小于 0")
    if settings["max_retries"] < 0:
        raise ValueError("max_retries 不能小于 0")
    if settings["retry_delay"] < 0:
        raise ValueError("retry_delay 不能小于 0")
    return settings


def _build_case(row: dict[str, Any], turns: list[str]) -> dict[str, Any]:
    """仅保留结果定位和分析需要的用例字段，跳过空值。"""
    case = {
        field: row[field]
        for field in CASE_FIELDS
        if row.get(field) is not None
    }
    case["turns"] = turns
    return case


def _warning_message(response_raw: dict[str, Any]) -> str:
    content = response_raw.get("content", {})
    data = content.get("data", {}) if isinstance(content, dict) else {}
    if isinstance(data, dict):
        for key in ("message", "description", "suggestion", "title"):
            value = data.get(key)
            if value:
                return str(value)
    if isinstance(content, dict) and content.get("title"):
        return str(content["title"])
    return "接口返回业务提示，未生成 confirmationId"


def _table_summary(blocks: list[dict[str, Any]]) -> dict[str, Any] | None:
    for block in blocks:
        if block.get("componentCode") != "data_table":
            continue
        data = block.get("data", {})
        if not isinstance(data, dict):
            return None
        columns = []
        raw_columns = data.get("columns") or []
        for column in raw_columns:
            if isinstance(column, dict):
                columns.append(
                    {
                        "key": column.get("key"),
                        "title": column.get("title"),
                        "data_type": column.get("dataType"),
                    }
                )
            else:
                columns.append(column)
        rows = data.get("rows") or []
        return {
            "columns": columns,
            "row_count": data.get("rowCount", len(rows)),
        }
    return None


def _extract_answer(agent: Any, response_raw: dict[str, Any]) -> dict[str, Any]:
    content = response_raw.get("content", {})
    blocks = (content.get("blocks") or []) if isinstance(content, dict) else []
    blocks = [block for block in blocks if isinstance(block, dict)]
    block_types = list(
        dict.fromkeys(
            block.get("componentCode")
            for block in blocks
            if block.get("componentCode")
        )
    )
    return {
        "text": agent._extract_summary(response_raw) or "",
        "block_types": block_types,
        "has_echart": "echart" in block_types,
        "table_summary": _table_summary(blocks),
    }


def _empty_turn_result(index: int, query: str) -> dict[str, Any]:
    return {
        "index": index,
        "query": query,
        "status": "request_failed",
        "duration_ms": 0,
        "routing": {
            "capability_id": None,
            "parameters": {},
        },
        "answer": {
            "text": "",
            "block_types": [],
            "has_echart": False,
            "table_summary": None,
        },
        "error": None,
        "_raw": {"requests": []},
    }


def _append_request_history(
    result: dict[str, Any],
    agent: Any,
    *,
    stage: str,
    attempt: int,
    fallback_response: Any = None,
    fallback_error: str | None = None,
) -> None:
    """记录子类暴露的 HTTP 尝试；普通 Agent 则记录逻辑调用结果。"""
    consume = getattr(agent, "consume_request_history", None)
    history = consume() if callable(consume) else []
    if history:
        for http_attempt in history:
            result["_raw"]["requests"].append(
                {
                    "stage": stage,
                    "attempt": attempt,
                    "auth_attempt": http_attempt.get("auth_attempt"),
                    "status_code": http_attempt.get("status_code"),
                    "response": http_attempt.get("response"),
                    "error": http_attempt.get("error"),
                }
            )
        return
    result["_raw"]["requests"].append(
        {
            "stage": stage,
            "attempt": attempt,
            "auth_attempt": 1,
            "status_code": None,
            "response": fallback_response,
            "error": fallback_error,
        }
    )


def _run_turn(
    agent: Any,
    query: str,
    *,
    session_id: str | None,
    max_retries: int,
    retry_delay: float,
    case_id: Any,
    turn_number: int,
) -> tuple[dict[str, Any], str | None]:
    """执行一轮两段式查询，返回精简结果和实际 session_id。"""
    result = _empty_turn_result(turn_number, query)
    for attempt in range(max_retries + 1):
        started_at = time.perf_counter()
        request_stage = "new_query"
        try:
            new_query_raw, confirmation_id, actual_session_id = agent.new_query(
                query, session_id=session_id
            )
            _append_request_history(
                result,
                agent,
                stage="new_query",
                attempt=attempt + 1,
                fallback_response=new_query_raw,
            )

            if confirmation_id is None:
                content = (
                    new_query_raw.get("content", {})
                    if isinstance(new_query_raw, dict)
                    else {}
                )
                is_warning = (
                    isinstance(content, dict)
                    and content.get("status") == "warning"
                )
                result["status"] = (
                    "business_warning" if is_warning else "new_query_failed"
                )
                result["error"] = (
                    _warning_message(new_query_raw)
                    if isinstance(new_query_raw, dict)
                    else "第一步响应中没有 confirmationId"
                )
                result["duration_ms"] = round(
                    (time.perf_counter() - started_at) * 1000
                )
                return result, actual_session_id

            request_stage = "confirm_execute"
            confirm_raw = agent.confirm_execute(
                confirmation_id, actual_session_id
            )
            _append_request_history(
                result,
                agent,
                stage="confirm_execute",
                attempt=attempt + 1,
                fallback_response=confirm_raw,
            )
            result["routing"] = {
                "capability_id": agent.extract_capability_id(confirm_raw) or None,
                "parameters": agent.extract_parameters(confirm_raw) or {},
            }
            result["answer"] = _extract_answer(agent, confirm_raw)
            result["duration_ms"] = round(
                (time.perf_counter() - started_at) * 1000
            )
            result["status"] = "success"
            return result, actual_session_id
        except Exception as exc:
            _append_request_history(
                result,
                agent,
                stage=request_stage,
                attempt=attempt + 1,
                fallback_error=str(exc),
            )
            result["duration_ms"] = round(
                (time.perf_counter() - started_at) * 1000
            )
            result["error"] = str(exc)
            if attempt < max_retries:
                logger.warning(
                    "[重试] id=%s，第 %s 轮，第 %s/%s 次，错误: %s",
                    case_id,
                    turn_number,
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(retry_delay)
                continue
            return result, session_id

    return result, session_id


def process_case(
    agent: Any,
    row: dict[str, Any],
    *,
    query_field: str | None,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """执行一条用例并返回归一化结果，不修改原始用例。"""
    turns = resolve_turns(row, query_field)
    case_id = row.get("id", "?")
    result: dict[str, Any] = {
        "case": _build_case(row, turns),
        "execution": {
            "status": "skipped",
            "duration_ms": 0,
            "session_id": None,
            "total_turns": len(turns),
            "completed_turns": 0,
            "error": None,
            "raw_ref": None,
        },
        "turns": [],
    }
    if not turns:
        result["execution"]["error"] = "未找到有效问题字段"
        logger.warning("[跳过] id=%s，未找到有效问题字段", case_id)
        return result

    case_started_at = time.perf_counter()
    session_id: str | None = None
    for turn_number, query in enumerate(turns, start=1):
        turn_result, session_id = _run_turn(
            agent,
            query,
            session_id=session_id,
            max_retries=max_retries,
            retry_delay=retry_delay,
            case_id=case_id,
            turn_number=turn_number,
        )
        result["turns"].append(turn_result)
        if turn_result["status"] == "success":
            result["execution"]["completed_turns"] += 1
            continue

        completed = result["execution"]["completed_turns"]
        result["execution"]["status"] = (
            "partial_success" if completed else turn_result["status"]
        )
        result["execution"]["error"] = (
            f"第 {turn_number} 轮未完成: {turn_result['error']}"
        )
        break
    else:
        result["execution"]["status"] = "success"

    result["execution"]["session_id"] = session_id
    result["execution"]["duration_ms"] = round(
        (time.perf_counter() - case_started_at) * 1000
    )
    logger.info(
        "[完成] id=%s，状态=%s，完成轮次=%s/%s",
        case_id,
        result["execution"]["status"],
        result["execution"]["completed_turns"],
        len(turns),
    )
    return result


def _process_job(
    agent: Any,
    job: dict[str, Any],
    *,
    query_field: str | None,
    max_retries: int,
    retry_delay: float,
    on_result: Callable[[dict[str, Any]], None] | None,
) -> None:
    try:
        result = process_case(
            agent,
            job["case"],
            query_field=query_field,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
    except Exception as exc:
        turns = resolve_turns(job["case"], query_field)
        result = {
            "case": _build_case(job["case"], turns),
            "execution": {
                "status": "request_failed",
                "duration_ms": 0,
                "session_id": None,
                "total_turns": len(turns),
                "completed_turns": 0,
                "error": f"用例处理异常: {exc}",
                "raw_ref": None,
            },
            "turns": [],
        }
        logger.exception("用例处理异常，id=%s", job["case"].get("id", "?"))
    if on_result is None:
        job["result"] = result
    else:
        on_result(result)


def run_cases(
    rows: list[dict[str, Any]],
    *,
    query_field: str | None = None,
    settings: dict[str, int | float] | None = None,
    on_result: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """批量执行问数测试；可在每条完成时立即投递给异步写入器。"""
    if not rows:
        return []

    settings = settings or resolve_run_settings()
    agent = create_refreshing_dataqa()
    jobs = [{"case": row, "result": None} for row in rows]
    handler = partial(
        _process_job,
        agent,
        query_field=query_field,
        max_retries=int(settings["max_retries"]),
        retry_delay=float(settings["retry_delay"]),
        on_result=on_result,
    )
    run_in_thread_pool(
        handler,
        jobs,
        max_workers=int(settings["max_workers"]),
        submit_delay=float(settings["submit_delay"]),
        task_name="DataQA 调度",
    )
    if on_result is not None:
        return []
    return [job["result"] for job in jobs if job["result"] is not None]


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    """按用例级状态汇总结果。"""
    summary: dict[str, int] = {"total": len(results)}
    for result in results:
        status = str(result.get("execution", {}).get("status") or "unknown")
        summary[status] = summary.get(status, 0) + 1
    return summary
