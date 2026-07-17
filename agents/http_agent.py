import requests
import json
import time
import uuid
from functools import wraps
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)


def _timing(func):
    """计时装饰器：将函数返回值扩展为 (*result, elapsed_seconds)"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return (*result, f'{elapsed:.4f}')
    return wrapper


class PVAssistant:
    """
    通过平台接口调用agent
    """
    def __init__(self, base_url:str, business_token:str):
        self.base_url = base_url

        self._base_header = {
            "X-Trace-Id": "0123456789abcdef0123456789abcdef",
            "X-System-Code": "agent-workbench",
            "X-Business-Token": str(business_token),
            "X-Platform": "web",
            "X-Tenant-Id": "146085512914162117",
            "Content-Type": "application/json"
        }

    @_timing
    def call_agent(self, question:str, user:str="lisanming-auto-test", 
                   session_id:str=None, res_mode:str='blocking', **kwargs)->list:
        """
        agent调用接口,返回agent的回答

        :param question: 用户输入的问题
        :param res_mode: agent响应方式：streaming|blocking
        :param user: 调用方标记
        :param session_id: 会话ID，用于维护上下文。
                          并发调用时必须传入唯一值以避免后端混淆响应；
                          默认自动生成 UUID。
        :param kwargs: 其他参数，为后续扩展预留
        :return: agent的回答,json解析后的字典
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        data = {
            "inputs": {},
            "query": question,
            "response_mode": res_mode,
            "user": user,
            "files": []
        }
        payload = json.dumps(data, ensure_ascii=False)

        # 每次请求复制 header，避免并发下的竞态条件
        headers = dict(self._base_header)
        headers['X-Session-Id'] = session_id
        response = requests.post(self.base_url, data=payload, headers=headers)
        if response.status_code != 200:
            logger.error(f"Agent调用失败，状态码: {response.status_code}, 响应内容: {response.text}, 实际请求：{response.request.body}")
            raise RuntimeError(f"Agent调用失败，状态码: {response.status_code}, 响应内容: {response.text}, 实际请求：{response.request.body}")

        # 格式化输出
        try:
            response_raw = response.json()
            response_summary = response_raw['content']['data']['markdown']
        except json.JSONDecodeError:
            logger.error(f"响应内容不是有效的JSON格式: {response.text}")
            raise RuntimeError(f"响应内容不是有效的JSON格式: {response.text}")
        else:
            return response_raw, response_summary
        
class Diagnosis:
    """
    诊断智能体，调用诊断推理API接口（默认 blocking 模式）
    """

    def __init__(self, base_url: str, business_token: str):
        self.base_url = base_url

        self._base_header = {
            "X-Trace-Id": "0123456789abcdef0123456789abcdef",
            "X-System-Code": "agent-workbench",
            "X-Business-Token": str(business_token),
            "X-Platform": "web",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    @_timing
    def call_agent(self, question: str, user: str = "lisanming-auto-test",
                   session_id: str = None, res_mode: str = "blocking", **kwargs):
        """
        调用诊断智能体

        :param question: 诊断问题，如 "故障码1616"
        :param user: 调用方用户标识
        :param session_id: 会话ID，默认自动生成 UUID
        :param res_mode: 响应模式，默认 blocking（streaming 暂未实现）
        :return: (response_raw, response_summary, res_time)
        """
        if res_mode != "blocking":
            raise NotImplementedError("Diagnosis 当前仅支持 blocking 模式")

        if session_id is None:
            session_id = str(uuid.uuid4())

        data = {
            "inputs": {},
            "query": question,
            "response_mode": res_mode,
            "user": user,
            "files": []
        }
        payload = json.dumps(data, ensure_ascii=False)

        headers = dict(self._base_header)
        headers['X-Session-Id'] = session_id

        response = requests.post(self.base_url, data=payload, headers=headers)
        if response.status_code != 200:
            logger.error(
                f"诊断Agent调用失败，状态码: {response.status_code}, "
                f"响应内容: {response.text}"
            )
            raise RuntimeError(
                f"诊断Agent调用失败，状态码: {response.status_code}, "
                f"响应内容: {response.text}"
            )

        try:
            response_raw = response.json()
            response_summary = response_raw['content']['data']['markdown']
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"诊断响应解析失败: {e}, 响应内容: {response.text}")
            raise RuntimeError(f"诊断响应解析失败: {e}")
        else:
            return response_raw, response_summary


class DataQA:
    """
    DataQA 问数智能体，通过 `/chat-messages` 接口完成数据查询。

    两步流程：
      1. new_query        — 发起全新查询，后端返回确认卡（含 confirmation_id）
      2. confirm_execute  — 复用 session_id 和 confirmation_id 执行确认查询
    """

    def __init__(self, base_url: str, business_token: str | None = None):
        self.base_url = base_url

        if business_token is None:
            from tool.business_platform_token_manager import ensure_valid_token
            business_token = ensure_valid_token()

        self._base_header = {
            "X-Trace-Id": "0123456789abcdef0123456789abcdef",
            "X-System-Code": "agent-workbench",
            "X-Business-Token": str(business_token),
            "X-Platform": "web",
            "X-Tenant-Id": "146085512914162117",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    # ------------------------------------------------------------------
    # 请求层
    # ------------------------------------------------------------------

    @_timing
    def call_agent(self, question: str, user: str = "lisanming-auto-test",
                   session_id: str = None, res_mode: str = "blocking",
                   inputs: dict = None, **kwargs) -> tuple:
        """
        HTTP 请求统一入口

        :param question: 用户自然语言问题
        :param user: 调用方用户标识
        :param session_id: 会话ID，默认自动生成 UUID（并发时须传入唯一值）
        :param res_mode: 响应模式，blocking 或 streaming
        :param inputs: 请求 payload 中的 inputs 字段
        :return: (response_raw, res_time) 原始 JSON 响应 + 耗时
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        data = {
            "inputs": inputs or {},
            "query": question,
            "response_mode": res_mode,
            "user": user,
            "files": []
        }
        payload = json.dumps(data, ensure_ascii=False)

        headers = dict(self._base_header)
        headers["X-Session-Id"] = session_id
        response = requests.post(self.base_url, data=payload, headers=headers)
        if response.status_code != 200:
            logger.error(
                f"DataQA调用失败，状态码: {response.status_code}, "
                f"响应内容: {response.text}"
            )
            raise RuntimeError(
                f"DataQA调用失败，状态码: {response.status_code}, "
                f"响应内容: {response.text}"
            )

        try:
            response_raw = response.json()
        except json.JSONDecodeError:
            logger.error(f"响应内容不是有效的JSON格式: {response.text}")
            raise RuntimeError(f"响应内容不是有效的JSON格式: {response.text}")

        return (response_raw,)

    # ------------------------------------------------------------------
    # 步骤方法
    # ------------------------------------------------------------------

    def new_query(self, question: str, user: str = "zhangsan",
                  session_id: str = None, res_mode: str = "blocking",
                  **kwargs) -> tuple:
        """
        步骤1 — 发起全新查询请求

        :param question: 用户自然语言问题
        :param user: 调用方用户标识
        :param session_id: 会话ID（并发时须传入唯一值）
        :param res_mode: 响应模式
        :param kwargs: 预留扩展参数（如 turn_intent="new_query" 强制新查询）
        :return: (response_raw, confirmation_id, session_id)
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        inputs = {}
        turn_intent = kwargs.get("turn_intent")
        if turn_intent:
            inputs["payload"] = {"turn_intent": turn_intent}

        response_raw, _ = self.call_agent(
            question=question, user=user, session_id=session_id,
            res_mode=res_mode, inputs=inputs,
        )
        confirmation_id = self._extract_confirmation_id(response_raw)

        return response_raw, confirmation_id, session_id

    def confirm_execute(self, confirmation_id: str, session_id: str,
                        user: str = "zhangsan", res_mode: str = "blocking",
                        query_text: str = "确认并查询") -> dict:
        """
        步骤2 — 确认执行查询，复用步骤1的 session_id

        :param confirmation_id: 步骤1 返回的确认卡 ID
        :param session_id: 必须与步骤1 使用相同的 session_id
        :param user: 调用方用户标识
        :param res_mode: 响应模式
        :param query_text: 确认查询的 query 文本
        :return: response_raw
        """
        if confirmation_id is None:
            return f"查询请求异常，检查{session_id}的new_qury请求响应"
        inputs = {
            "payload": {
                "turn_intent": "confirm_execute",
                "confirmation_id": confirmation_id,
            }
        }
        response_raw, _ = self.call_agent(
            question=query_text, user=user, session_id=session_id,
            res_mode=res_mode, inputs=inputs,
        )
        return response_raw

    # ------------------------------------------------------------------
    # 响应解析工具
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_confirmation_id(response_raw: dict) -> str | None:
        """从 DataQA step-1 响应中提取 confirmation_id"""
        try:
            content = response_raw.get("content", {})
            cid = content.get("data", {}).get("confirmationId")
            if cid:
                return cid
        except Exception:
            logger.debug("未匹配到confirmationId")
            logger.debug(f"{response_raw}")
            pass
        return None

    @staticmethod
    def extract_capability_id(response_raw: dict) -> str:
        """从 DataQA step-2 响应中提取实际执行的 capability_id

        :param response_raw: confirm_execute 返回的原始 JSON 字典
        :return: capability_id 字符串，如 "work_order_list"，提取失败返回 ""
        """
        try:
            return response_raw.get("content", {}).get("metadata", {}).get("capability_id", "")
        except Exception:
            return ""

    @staticmethod
    def extract_parameters(response_raw: dict) -> dict:
        """从 DataQA step-2 响应中提取实际查询参数

        :param response_raw: confirm_execute 返回的原始 JSON 字典
        :return: 参数字典，如 {"siteId": "xxx", "startTime": "2026-07-05 00:00:00"}，
                 提取失败返回 {}
        """
        try:
            return response_raw.get("content", {}).get("metadata", {}).get("params", {})
        except Exception:
            return {}

    @staticmethod
    def _extract_summary(response_raw: dict) -> str:
        """从 DataQA step-2 响应中提取 markdown 文本摘要

        优先从 content.blocks[] 中 markdown_viewer 的 markdown 取值；
        其次尝试 content.data.markdown（兼容旧格式）；
        都没有则返回空字符串。

        :param response_raw: confirm_execute 返回的原始 JSON 字典
        :return: 文本摘要字符串
        """
        try:
            content = response_raw.get("content", {})
            blocks = content.get("blocks", [])
            for block in blocks:
                if block.get("componentCode") == "markdown_viewer":
                    md = block.get("data", {}).get("markdown", "")
                    if md:
                        return md
            fallback = content.get("data", {}).get("markdown", "")
            if fallback:
                return fallback
            return ""
        except Exception:
            return ""


if __name__ == "__main__":
    agent = PVAssistant(base_url="http://192.168.100.189:32245/chat-messages",
                        business_token="11")
    question = "集中式逆变器发生故障时，对电站发电量有何影响？"
    answer_raw, answer_summary, res_time = agent.call_agent(question)
    print("Agent的回答:", answer_raw)
    print("ageng回答的summary:", answer_summary)
    print('响应时间:', res_time)
    print(type(answer_raw))