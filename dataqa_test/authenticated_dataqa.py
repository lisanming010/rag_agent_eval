"""临时问数测试专用的独立鉴权管理和 DataQA 子类。"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import requests
import yaml

from agents.http_agent import DataQA
from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)

DEFAULT_LOGIN_URL = "https://intelligent-bi.codeartz.xyz/api/system/login"
DEFAULT_USERNAME = "Leo01"
DEFAULT_TENANT_ID = "146085512914162117"
DEFAULT_LANGUAGE = "en-US"
DEFAULT_AUTH_CONFIG = Path(__file__).resolve().parent / "auth.yaml"

PostCallable = Callable[..., Any]


def _timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        started_at = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - started_at
        return (*result, f"{elapsed:.4f}")

    return wrapper


@dataclass(frozen=True)
class AuthSettings:
    login_url: str
    username: str
    password: str
    tenant_id: str
    accept_language: str
    login_timeout: float
    request_timeout: float
    agent_endpoint: str | None

    @classmethod
    def load(cls, config_path: Path | None = None) -> "AuthSettings":
        """从独立 YAML 读取配置，环境变量具有最高优先级。"""
        path = config_path or DEFAULT_AUTH_CONFIG
        config: dict[str, Any] = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                loaded = yaml.safe_load(file) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"鉴权配置必须为 YAML 对象: {path}")
            config = loaded

        def pick(env_name: str, key: str, default: Any = None) -> Any:
            env_value = os.getenv(env_name)
            return env_value if env_value is not None else config.get(key, default)

        password = str(pick("DATAQA_AUTH_PASSWORD", "password", "")).strip()
        if not password:
            raise ValueError(
                "未配置问数鉴权密码，请创建 dataqa_test/auth.yaml 或设置 "
                "DATAQA_AUTH_PASSWORD"
            )

        endpoint = pick("DATAQA_AGENT_ENDPOINT", "agent_endpoint")
        return cls(
            login_url=str(
                pick("DATAQA_AUTH_LOGIN_URL", "login_url", DEFAULT_LOGIN_URL)
            ).strip(),
            username=str(
                pick("DATAQA_AUTH_USERNAME", "username", DEFAULT_USERNAME)
            ).strip(),
            password=password,
            tenant_id=str(
                pick("DATAQA_AUTH_TENANT_ID", "tenant_id", DEFAULT_TENANT_ID)
            ).strip(),
            accept_language=str(
                pick(
                    "DATAQA_AUTH_ACCEPT_LANGUAGE",
                    "accept_language",
                    DEFAULT_LANGUAGE,
                )
            ).strip(),
            login_timeout=float(
                pick("DATAQA_AUTH_LOGIN_TIMEOUT", "login_timeout", 30)
            ),
            request_timeout=float(
                pick("DATAQA_REQUEST_TIMEOUT", "request_timeout", 120)
            ),
            agent_endpoint=str(endpoint).strip() if endpoint else None,
        )


class BusinessAuthTokenManager:
    """线程安全的内存 Token 缓存；JWT 过期或 401 时重新登录。"""

    def __init__(
        self,
        settings: AuthSettings,
        *,
        post: PostCallable = requests.post,
        expiry_skew_seconds: int = 30,
    ):
        self.settings = settings
        self._post = post
        self._expiry_skew_seconds = expiry_skew_seconds
        self._token: str | None = None
        self._expires_at: float | None = None
        self._lock = threading.Lock()

    def get_token(self) -> str:
        """返回可用 token；无缓存或即将过期时只允许一个线程登录。"""
        if self._is_cached_token_valid():
            return self._token  # type: ignore[return-value]
        with self._lock:
            if self._is_cached_token_valid():
                return self._token  # type: ignore[return-value]
            return self._login_locked()

    def refresh_after_unauthorized(self, failed_token: str) -> str:
        """401 后刷新；若其他线程已经刷新，则直接复用新 token。"""
        with self._lock:
            if self._token and self._token != failed_token:
                return self._token
            return self._login_locked()

    def _login_locked(self) -> str:
        response = self._post(
            self.settings.login_url,
            json={
                "username": self.settings.username,
                "password": self.settings.password,
            },
            headers={
                "Content-Type": "application/json",
                "Accept-Language": self.settings.accept_language,
            },
            timeout=self.settings.login_timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(f"业务平台登录失败，状态码: {response.status_code}")
        try:
            payload = response.json()
            token = payload["data"]["access_token"]
        except (ValueError, KeyError, TypeError) as exc:
            raise RuntimeError("业务平台登录响应缺少 data.access_token") from exc
        if not isinstance(token, str) or not token.strip():
            raise RuntimeError("业务平台登录响应中的 access_token 为空")

        self._token = token.strip()
        self._expires_at = self._decode_jwt_exp(self._token)
        logger.info("业务平台鉴权 token 已刷新")
        return self._token

    def _is_cached_token_valid(self) -> bool:
        if not self._token:
            return False
        if self._expires_at is None:
            return True
        return time.time() + self._expiry_skew_seconds < self._expires_at

    @staticmethod
    def _decode_jwt_exp(token: str) -> float | None:
        """仅解析 JWT exp 用于缓存判断，不进行签名验证。"""
        try:
            encoded_payload = token.split(".")[1]
            padding = "=" * (-len(encoded_payload) % 4)
            decoded = base64.urlsafe_b64decode(encoded_payload + padding)
            exp = json.loads(decoded.decode("utf-8")).get("exp")
            return float(exp) if exp is not None else None
        except (
            IndexError,
            ValueError,
            TypeError,
            json.JSONDecodeError,
            binascii.Error,
        ):
            return None


class RefreshingDataQA(DataQA):
    """继承原 DataQA，仅覆盖 HTTP 请求以使用独立鉴权和 401 刷新。"""

    def __init__(
        self,
        base_url: str,
        token_manager: BusinessAuthTokenManager,
        *,
        tenant_id: str = DEFAULT_TENANT_ID,
        accept_language: str = DEFAULT_LANGUAGE,
        request_timeout: float = 120,
        post: PostCallable = requests.post,
    ):
        # 传入占位值，避免触发原有业务平台 TokenManager。
        super().__init__(base_url=base_url, business_token="managed-by-dataqa-test")
        self._token_manager = token_manager
        self._request_timeout = request_timeout
        self._post = post
        self._request_state = threading.local()
        self._base_header.update(
            {
                "X-Tenant-Id": tenant_id,
                "Accept-Language": accept_language,
            }
        )

    def consume_request_history(self) -> list[dict[str, Any]]:
        """取出当前线程最近一次逻辑调用中的 HTTP 尝试记录。"""
        history = getattr(self._request_state, "history", [])
        self._request_state.history = []
        return history

    @_timing
    def call_agent(
        self,
        question: str,
        user: str = "lisanming-auto-test",
        session_id: str = None,
        res_mode: str = "blocking",
        inputs: dict = None,
        **kwargs,
    ) -> tuple:
        """使用 JWT 请求问数接口；401 时刷新 token 并重试一次。"""
        if session_id is None:
            session_id = str(uuid.uuid4())
        payload = json.dumps(
            {
                "inputs": inputs or {},
                "query": question,
                "response_mode": res_mode,
                "user": user,
                "files": [],
            },
            ensure_ascii=False,
        )

        token = self._token_manager.get_token()
        history: list[dict[str, Any]] = []
        self._request_state.history = history
        for auth_attempt in range(2):
            headers = dict(self._base_header)
            headers.update(
                {
                    "Authorization": f"Bearer {token}",
                    "X-Business-Token": token,
                    "X-Session-Id": session_id,
                }
            )
            try:
                response = self._post(
                    self.base_url,
                    data=payload,
                    headers=headers,
                    timeout=self._request_timeout,
                )
            except Exception as exc:
                history.append(
                    {
                        "auth_attempt": auth_attempt + 1,
                        "status_code": None,
                        "response": None,
                        "error": str(exc),
                    }
                )
                raise
            response_json_valid = True
            try:
                response_payload = response.json()
            except ValueError:
                response_json_valid = False
                response_payload = {"text": response.text}
            history.append(
                {
                    "auth_attempt": auth_attempt + 1,
                    "status_code": response.status_code,
                    "response": response_payload,
                    "error": None,
                }
            )
            if response.status_code == 401 and auth_attempt == 0:
                logger.warning("DataQA 返回 401，刷新 token 后重试一次")
                token = self._token_manager.refresh_after_unauthorized(token)
                continue
            if response.status_code == 401:
                raise RuntimeError("DataQA 鉴权失败，刷新 token 后仍返回 401")
            if response.status_code != 200:
                raise RuntimeError(
                    f"DataQA 调用失败，状态码: {response.status_code}"
                )
            if not response_json_valid or not isinstance(response_payload, dict):
                raise RuntimeError("DataQA 响应 JSON 顶层不是对象")
            return (response_payload,)

        raise RuntimeError("DataQA 鉴权重试流程异常结束")


def create_refreshing_dataqa(
    config_path: Path | None = None,
    *,
    auth_post: PostCallable = requests.post,
    dataqa_post: PostCallable = requests.post,
) -> RefreshingDataQA:
    """创建临时测试专用 DataQA，不经过原 Agent 工厂的鉴权路线。"""
    settings = AuthSettings.load(config_path)
    endpoint = settings.agent_endpoint or ConfigReader.get_instance().get(
        "agents.http_agent.class_config.DataQA.endpoint"
    )
    if not endpoint:
        raise ValueError(
            "未配置 DataQA endpoint，请在 dataqa_test/auth.yaml 设置 "
            "agent_endpoint，或配置原项目 DataQA endpoint"
        )
    token_manager = BusinessAuthTokenManager(settings, post=auth_post)
    return RefreshingDataQA(
        base_url=str(endpoint),
        token_manager=token_manager,
        tenant_id=settings.tenant_id,
        accept_language=settings.accept_language,
        request_timeout=settings.request_timeout,
        post=dataqa_post,
    )
