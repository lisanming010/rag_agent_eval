import requests
import json
import os
import time
import uuid
from functools import wraps
from typing import Optional

from dotenv import load_dotenv

from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory

load_dotenv()
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
            # "X-Tenant-Id": "146085512914162117",
            "X-Tenant-Id": "1",
            "Content-Type": "application/json"
        }

    @_timing
    def call_agent(self, question:str, user:str="lisanming-auto-test",
                   session_id:str=None, res_mode:str='blocking',
                   tenant_id:str=None, **kwargs)->list:
        """
        agent调用接口,返回agent的回答

        :param question: 用户输入的问题
        :param res_mode: agent响应方式：streaming|blocking
        :param user: 调用方标记
        :param session_id: 会话ID，用于维护上下文。
                          并发调用时必须传入唯一值以避免后端混淆响应；
                          默认自动生成 UUID。
        :param tenant_id: 租户ID，传入时覆盖 _base_header 中的 X-Tenant-Id
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
        if tenant_id is not None:
            headers['X-Tenant-Id'] = tenant_id
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
    诊断智能体，调用诊断推理 API 接口（默认 SSE streaming 模式）
    """

    DEFAULT_LANGUAGE = "zh-CN"
    SUPPORTED_LANGUAGES = frozenset({"zh-CN", "en-US"})
    DEFAULT_RESPONSE_MODE = "streaming"
    SUPPORTED_RESPONSE_MODES = frozenset({"blocking", "streaming"})

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
                   session_id: str = None, res_mode: str = DEFAULT_RESPONSE_MODE,
                   language: str = DEFAULT_LANGUAGE, **kwargs):
        """
        调用诊断智能体

        :param question: 诊断问题，如 "故障码1616"
        :param user: 调用方用户标识
        :param session_id: 会话ID，默认自动生成 UUID
        :param res_mode: 响应模式，默认 streaming；兼容后端返回的 SSE/JSON
        :param language: 响应语言请求头，仅支持 zh-CN / en-US，默认 zh-CN
        :return: (response_raw, response_summary, res_time)
        """
        if res_mode not in self.SUPPORTED_RESPONSE_MODES:
            supported_modes = ", ".join(sorted(self.SUPPORTED_RESPONSE_MODES))
            raise ValueError(
                f"Diagnosis res_mode 仅支持 {supported_modes}，实际值: {res_mode!r}"
            )

        language = str(language or self.DEFAULT_LANGUAGE).strip()
        if language not in self.SUPPORTED_LANGUAGES:
            supported = ", ".join(sorted(self.SUPPORTED_LANGUAGES))
            raise ValueError(
                f"Diagnosis language 仅支持 {supported}，实际值: {language!r}"
            )

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
        headers['Language'] = language
        if res_mode == "streaming":
            headers['Accept'] = 'text/event-stream'

        response = requests.post(
            self.base_url,
            data=payload,
            headers=headers,
            stream=res_mode == "streaming",
        )
        try:
            if response.status_code != 200:
                error_message = (
                    f"诊断Agent调用失败，状态码: {response.status_code}, "
                    f"响应内容: {response.text}"
                )
                logger.error(error_message)
                raise RuntimeError(error_message)
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/event-stream' in content_type:
                return self._parse_sse_response(response)
            return self._parse_json_response(response)
        except RuntimeError:
            raise
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"诊断响应解析失败: {e}")
            raise RuntimeError(f"诊断响应解析失败: {e}") from e
        finally:
            close = getattr(response, 'close', None)
            if callable(close):
                close()

    @staticmethod
    def _parse_json_response(response):
        """解析兼容的非流式 JSON 响应。"""
        response_raw = response.json()
        response_summary = response_raw['content']['data']['markdown']
        if not isinstance(response_summary, str) or not response_summary.strip():
            raise ValueError("JSON 响应中的 markdown 内容为空")
        return response_raw, response_summary

    @staticmethod
    def _parse_sse_response(response):
        """解析诊断接口 SSE，优先提取 complete.data.content。"""
        complete_event = None
        delta_parts = []
        event_count = 0

        for raw_line in response.iter_lines(decode_unicode=True):
            if isinstance(raw_line, bytes):
                line = raw_line.decode('utf-8')
            else:
                line = str(raw_line or '')
            line = line.strip()
            if not line or line.startswith(':') or not line.startswith('data:'):
                continue

            event_payload = line[len('data:'):].strip()
            if not event_payload or event_payload == '[DONE]':
                continue

            event = json.loads(event_payload)
            event_count += 1
            event_type = event.get('event_type') or event.get('eventType')
            event_data = event.get('data') or {}

            if event_type == 'delta':
                delta = event_data.get('contentDelta') or event_data.get('content')
                if isinstance(delta, str):
                    delta_parts.append(delta)
            elif event_type == 'complete':
                if event_data.get('status') not in (None, 'SUCCESS'):
                    raise ValueError(
                        f"SSE complete 状态异常: {event_data.get('status')!r}"
                    )
                content = event_data.get('content')
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("SSE complete 事件缺少 data.content")
                complete_event = event
            elif event_type == 'error':
                raise ValueError(f"SSE error 事件: {event_data}")

        if complete_event is not None:
            return complete_event, complete_event['data']['content']

        if delta_parts:
            response_summary = ''.join(delta_parts)
            response_raw = {
                'event_type': 'complete',
                'data': {
                    'status': 'SUCCESS',
                    'content': response_summary,
                    'source': 'delta_fallback',
                },
            }
            return response_raw, response_summary

        raise ValueError(f"SSE 响应未包含 complete 或 delta 内容（事件数: {event_count}）")


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


class RAGFlowRetriever:
    """
    RAGFlow 知识库检索器，通过 REST API 从指定知识库中检索相关 chunks。

    配置来源（config.yaml 中 agents.http_agent.ragflow 段）：
      - api_url: RAGFlow API 地址
      - dataset_id: 默认数据集 ID
      - top_k / similarity_threshold / vector_similarity_weight: 检索参数
      - api_key: 从 .env 的 RAGFLOW_API_KEY 环境变量读取

    使用示例::

        retriever = RAGFlowRetriever()
        result = retriever.retrieve(question="什么是RAGFlow?")
        for chunk in result["chunks"]:
            print(chunk["content"], chunk["similarity"])
    """

    def __init__(
        self,
        api_url: str | None = None,
        api_key: str | None = None,
        dataset_id: str | None = None,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        vector_similarity_weight: float | None = None,
    ):
        """
        :param api_url: RAGFlow API 地址，默认从 config 读取
        :param api_key: RAGFlow API Key，默认从环境变量 RAGFLOW_API_KEY 读取
        :param dataset_id: 默认数据集 ID，默认从 config 读取
        :param top_k: 向量余弦计算涉及的 chunk 数量，默认 1024
        :param similarity_threshold: 最小相似度阈值，默认 0.5
        :param vector_similarity_weight: 向量余弦相似度权重，默认 0.7
        """
        conf = ConfigReader.get_instance()
        ragflow_conf = conf.get("agents.http_agent.ragflow", {})

        self.api_url = (
            api_url
            or ragflow_conf.get("api_url", "http://192.168.100.225")
        ).rstrip("/")

        self.api_key = api_key or os.getenv("RAGFLOW_API_KEY", "")

        self.dataset_id = (
            dataset_id if dataset_id is not None
            else ragflow_conf.get("dataset_id", "")
        )

        self.top_k = (
            top_k if top_k is not None
            else ragflow_conf.get("top_k", 1024)
        )
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None
            else ragflow_conf.get("similarity_threshold", 0.5)
        )
        self.vector_similarity_weight = (
            vector_similarity_weight if vector_similarity_weight is not None
            else ragflow_conf.get("vector_similarity_weight", 0.7)
        )

        self._retrieval_url = f"{self.api_url}/api/v1/retrieval"

    # ------------------------------------------------------------------
    # 核心检索方法
    # ------------------------------------------------------------------

    @_timing
    def retrieve(
        self,
        question: str,
        dataset_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        page: int = 1,
        page_size: int = 30,
        top_k: int | None = None,
        similarity_threshold: float | None = None,
        vector_similarity_weight: float | None = None,
        keyword: bool = False,
        highlight: bool = False,
        use_kg: bool = False,
        toc_enhance: bool = False,
        **kwargs,
    ) -> tuple[dict, list[dict]]:
        """
        调用 RAGFlow 检索接口，返回原始响应与 chunk 列表。

        若未传入 dataset_ids 且未传入 document_ids，自动使用实例的 dataset_id。

        :param question: 查询问题 / 关键词
        :param dataset_ids: 数据集 ID 列表，不传则使用实例默认 dataset_id
        :param document_ids: 文档 ID 列表，与 dataset_ids 二选一
        :param page: 分页页码，默认 1
        :param page_size: 每页最大 chunk 数，默认 30
        :param top_k: 覆盖实例级别的 top_k
        :param similarity_threshold: 覆盖实例级别的 similarity_threshold
        :param vector_similarity_weight: 覆盖实例级别的 vector_similarity_weight
        :param keyword: 是否启用关键词匹配
        :param highlight: 是否高亮匹配词
        :param use_kg: 是否使用知识图谱进行多跳查询
        :param toc_enhance: 是否使用目录增强检索
        :param kwargs: 其他可选参数，传递到请求 body
        :return: (response_raw, chunks)
        :raises RuntimeError: API 调用失败或响应解析失败时抛出
        """
        if not self.api_key:
            raise RuntimeError(
                "RAGFLOW_API_KEY 未配置，请在 .env 文件中设置 RAGFLOW_API_KEY"
            )

        # 未传 dataset_ids / document_ids 时回退到实例默认 dataset_id
        resolved_dataset_ids = dataset_ids
        if not resolved_dataset_ids and not document_ids and self.dataset_id:
            resolved_dataset_ids = [self.dataset_id]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        body: dict = {
            "question": question,
            "page": page,
            "page_size": page_size,
            "similarity_threshold": (
                similarity_threshold if similarity_threshold is not None
                else self.similarity_threshold
            ),
            "vector_similarity_weight": (
                vector_similarity_weight if vector_similarity_weight is not None
                else self.vector_similarity_weight
            ),
            "top_k": top_k if top_k is not None else self.top_k,
            "keyword": keyword,
            "highlight": highlight,
            "use_kg": use_kg,
            "toc_enhance": toc_enhance,
        }

        if resolved_dataset_ids:
            body["dataset_ids"] = resolved_dataset_ids
        if document_ids:
            body["document_ids"] = document_ids

        body.update(kwargs)

        logger.info(
            f"RAGFlow 检索请求: question={question[:80]}..., "
            f"dataset_ids={resolved_dataset_ids}, top_k={body['top_k']}"
        )

        response = requests.post(
            self._retrieval_url,
            data=json.dumps(body, ensure_ascii=False),
            headers=headers,
        )

        if response.status_code != 200:
            logger.error(
                f"RAGFlow 检索失败，状态码: {response.status_code}, "
                f"响应内容: {response.text}"
            )
            raise RuntimeError(
                f"RAGFlow 检索失败，状态码: {response.status_code}, "
                f"响应内容: {response.text}"
            )

        try:
            response_raw = response.json()
        except json.JSONDecodeError:
            logger.error(f"RAGFlow 响应不是有效的 JSON: {response.text}")
            raise RuntimeError(f"RAGFlow 响应不是有效的 JSON: {response.text}")

        if response_raw.get("code") != 0:
            error_msg = response_raw.get("message", "未知错误")
            logger.error(
                f"RAGFlow API 返回错误码: {response_raw.get('code')}, "
                f"消息: {error_msg}"
            )
            raise RuntimeError(f"RAGFlow API 错误: {error_msg}")

        chunks = (response_raw.get("data") or {}).get("chunks", [])

        return response_raw, chunks

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def retrieve_contents(
        self,
        question: str,
        dataset_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        min_similarity: float = 0.0,
        **kwargs,
    ) -> list[str]:
        """
        便捷方法：检索并返回 chunk 文本内容列表。

        :param question: 查询问题
        :param dataset_ids: 数据集 ID 列表
        :param document_ids: 文档 ID 列表
        :param min_similarity: 最低相似度过滤阈值，默认 0.0 不过滤
        :param kwargs: 传递到 retrieve() 的其他参数
        :return: chunk 文本内容列表
        """
        _, chunks = self.retrieve(
            question=question,
            dataset_ids=dataset_ids,
            document_ids=document_ids,
            **kwargs,
        )
        return [
            c["content"]
            for c in chunks
            if c.get("similarity", 0) >= min_similarity
        ]

    def retrieve_with_scores(
        self,
        question: str,
        dataset_ids: list[str] | None = None,
        document_ids: list[str] | None = None,
        **kwargs,
    ) -> list[dict]:
        """
        便捷方法：检索并返回带相似度分数的 chunk 摘要列表。

        :param question: 查询问题
        :param dataset_ids: 数据集 ID 列表
        :param document_ids: 文档 ID 列表
        :param kwargs: 传递到 retrieve() 的其他参数
        :return: [{content, similarity, vector_similarity, term_similarity,
                   document_name, document_id, chunk_id}, ...]
        """
        _, chunks = self.retrieve(
            question=question,
            dataset_ids=dataset_ids,
            document_ids=document_ids,
            **kwargs,
        )
        return [
            {
                "content": c.get("content", ""),
                "similarity": c.get("similarity"),
                "vector_similarity": c.get("vector_similarity"),
                "term_similarity": c.get("term_similarity"),
                "document_name": c.get("document_keyword", ""),
                "document_id": c.get("document_id", ""),
                "chunk_id": c.get("id", ""),
            }
            for c in chunks
        ]


if __name__ == "__main__":
    agent = PVAssistant(base_url="http://192.168.100.189:32245/chat-messages",
                        business_token="11")
    question = "集中式逆变器发生故障时，对电站发电量有何影响？"
    answer_raw, answer_summary, res_time = agent.call_agent(question)
    print("Agent的回答:", answer_raw)
    print("ageng回答的summary:", answer_summary)
    print('响应时间:', res_time)
    print(type(answer_raw))
