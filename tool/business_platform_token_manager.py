"""
业务平台 Token 管理工具

通过用户名 / 密码请求登录 API 获取 JWT access_token，
持久化到本地文件，供后续 API 调用复用；过期自动重新登录。

配置项位于 config.yaml 的 business_platform 

典型用法：
    from tool.business_platform_token_manager import BusinessPlatformTokenManager

    tm = BusinessPlatformTokenManager()
    token = tm.get_valid_token()
    headers = {"Authorization": f"Bearer {token}"}

命令行用法：
    python -m tool.business_platform_token_manager          # 登录并保存 token
    python -m tool.business_platform_token_manager --token   # 仅打印当前有效 token
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from tool.config_reader import ConfigReader

# ---------- 从配置文件读取默认值 ----------
_cfg = ConfigReader()

DEFAULT_LOGIN_URL = _cfg.get("business_platform.login_url", "")
DEFAULT_USERNAME = _cfg.get("business_platform.username", "")
DEFAULT_PASSWORD = _cfg.get("business_platform.password", "")

# 状态文件保存到项目根目录
DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent / "business_platform_auth_state.json"

# API 返回的 expires_in 仅为名义值（720s），JWT 无 exp 声明，服务端实际 session 长得多
# 此处使用较长的 fallback TTL，避免频繁重新登录
DEFAULT_FALLBACK_TTL_SECONDS = 3 * 60 * 60  # 3 小时

# 过期安全余量（秒），避免在临界点失败
EXPIRY_SAFETY_MARGIN_SECONDS = 300  # 5 分钟


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    """解码 JWT payload（不验证签名）"""
    try:
        parts = token.split(".")
        if len(parts) < 2:
            return {}
        # Base64 URL-decode 第二个部分（payload）
        payload = parts[1]
        payload += "=" * (4 - len(payload) % 4)
        import base64
        decoded = base64.urlsafe_b64decode(payload)
        return json.loads(decoded)
    except Exception:
        return {}


@dataclass
class PlatformTokenState:
    """本地持久化的 Token 状态"""

    login_url: str
    username: str
    saved_at: float
    expires_at: float
    access_token: str = ""
    expires_in: int = 720
    user_id: str = ""
    user_key: str = ""
    tenant_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "login_url": self.login_url,
            "username": self.username,
            "saved_at": self.saved_at,
            "expires_at": self.expires_at,
            "access_token": self.access_token,
            "expires_in": self.expires_in,
            "user_id": self.user_id,
            "user_key": self.user_key,
            "tenant_id": self.tenant_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlatformTokenState":
        return cls(
            login_url=data.get("login_url", ""),
            username=data.get("username", ""),
            saved_at=float(data.get("saved_at", 0)),
            expires_at=float(data.get("expires_at", 0)),
            access_token=data.get("access_token", ""),
            expires_in=int(data.get("expires_in", 720)),
            user_id=data.get("user_id", ""),
            user_key=data.get("user_key", ""),
            tenant_id=data.get("tenant_id", ""),
        )


class BusinessPlatformTokenManager:
    """业务平台 Token 管理器

    核心职责：
        1. 通过账号密码请求登录 API 获取 JWT
        2. 持久化到本地 JSON 文件
        3. 自动判断过期，过期后重新登录
        4. 提供便捷的 HTTP 鉴权头
    """

    def __init__(
        self,
        login_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        state_file: str | Path | None = None,
    ) -> None:
        self.login_url = login_url or DEFAULT_LOGIN_URL
        self.username = username or DEFAULT_USERNAME
        self.password = password or DEFAULT_PASSWORD
        self.state_file = Path(state_file) if state_file else DEFAULT_STATE_FILE

    # ---------- 持久化 ----------
    def load(self) -> PlatformTokenState | None:
        """从本地文件加载 token 状态"""
        if not self.state_file.exists():
            return None
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return PlatformTokenState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _save(self, state: PlatformTokenState) -> None:
        """保存 token 状态到本地文件（原子写入）"""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_file)

    def clear(self) -> None:
        """删除本地 token 状态"""
        if self.state_file.exists():
            self.state_file.unlink()

    # ---------- 过期判断 ----------
    def is_expired(self, state: PlatformTokenState | None = None) -> bool:
        """判断 token 是否已过期（含安全余量）"""
        state = state or self.load()
        if state is None:
            return True
        return time.time() >= (state.expires_at - EXPIRY_SAFETY_MARGIN_SECONDS)

    def expires_in_seconds(self, state: PlatformTokenState | None = None) -> float:
        """返回距离过期的剩余秒数"""
        state = state or self.load()
        if state is None:
            return 0.0
        return max(0.0, state.expires_at - time.time())

    # ---------- 核心：登录 ----------
    def login(self) -> PlatformTokenState:
        """使用账号密码登录，获取 access_token 并持久化

        POST /api/system/login
        Body: {"grant_type":"password","scope":"server","username":"...","password":"...","mode":"jwt"}

        Raises:
            RuntimeError: 登录失败
        """
        payload = {
            "grant_type": "password",
            "scope": "server",
            "username": self.username,
            "password": self.password,
            "mode": "jwt",
        }

        try:
            resp = requests.post(
                self.login_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30,
                verify=True,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"登录请求失败: {e}") from e
        except json.JSONDecodeError as e:
            raise RuntimeError(f"登录响应解析失败: {e}") from e

        code = body.get("code")
        if code != 200:
            msg = body.get("msg", "未知错误")
            raise RuntimeError(f"登录失败: [{code}] {msg}")

        data = body.get("data", {})
        access_token = data.get("access_token", "")
        if not access_token:
            raise RuntimeError("登录响应中未包含 access_token")

        # 从 JWT 中解析用户信息
        jwt_payload = _decode_jwt_payload(access_token)

        now = time.time()
        # API 返回的 expires_in 仅为名义值，使用 fallback 作为实际 TTL
        api_expires_in = int(data.get("expires_in", 720))
        expires_in = max(api_expires_in, DEFAULT_FALLBACK_TTL_SECONDS)

        state = PlatformTokenState(
            login_url=self.login_url,
            username=self.username,
            saved_at=now,
            expires_at=now + expires_in,
            access_token=access_token,
            expires_in=expires_in,
            user_id=str(jwt_payload.get("user_id", "")),
            user_key=str(jwt_payload.get("user_key", "")),
            tenant_id=str(jwt_payload.get("tenantId", "")),
        )
        self._save(state)

        print(f"[business_platform] 登录成功，token 有效期 {expires_in}s（约 {expires_in // 60} 分钟）")
        if state.user_id:
            print(f"[business_platform] user_id: {state.user_id}, tenant: {state.tenant_id}")
        return state

    def get_valid_token(self) -> str:
        """获取当前有效的 access_token，账号变更或过期自动重新登录

        Returns:
            str: 有效的 access_token

        Raises:
            RuntimeError: 无法获取有效 token
        """
        state = self.load()
        if state is None:
            state = self.login()
        elif state.username != self.username:
            print(f"[business_platform] 账号已变更 ({state.username} → {self.username})，重新登录...")
            state = self.login()
        elif self.is_expired(state):
            print("[business_platform] token 已过期，重新登录...")
            state = self.login()

        if not state.access_token:
            raise RuntimeError("无法获取有效的 access_token")

        return state.access_token

    # ---------- 便捷方法 ----------
    def get_auth_header(self) -> dict[str, str]:
        """获取 Authorization 请求头"""
        token = self.get_valid_token()
        return {"Authorization": f"Bearer {token}"}

    def get_cookie_header(self) -> str:
        """获取 Admin-Token cookie 字符串"""
        token = self.get_valid_token()
        return f"Admin-Token={token}"

    def get_state(self) -> PlatformTokenState | None:
        """获取当前持久化状态（不自动刷新）"""
        return self.load()


def ensure_valid_token(
    login_url: str | None = None,
    username: str | None = None,
    password: str | None = None,
    state_file: str | Path | None = None,
) -> str:
    """便捷函数：确保获取到有效 token，自动处理登录/过期"""
    tm = BusinessPlatformTokenManager(
        login_url=login_url,
        username=username,
        password=password,
        state_file=state_file,
    )
    return tm.get_valid_token()


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="业务平台 Token 管理工具")
    parser.add_argument("--token", action="store_true", help="仅打印当前有效 access_token 并退出")
    parser.add_argument("--clear", action="store_true", help="清除本地保存的 token 状态")
    parser.add_argument("--status", action="store_true", help="显示当前 token 状态信息")
    parser.add_argument("--username", default=None, help="登录用户名（默认使用 config.yaml 配置）")
    parser.add_argument("--password", default=None, help="登录密码（默认使用 config.yaml 配置）")

    args = parser.parse_args()

    tm = BusinessPlatformTokenManager(
        username=args.username,
        password=args.password,
    )

    if args.clear:
        tm.clear()
        print("[business_platform] token 状态已清除")
        sys.exit(0)

    if args.token:
        try:
            token = tm.get_valid_token()
            print(token)
        except RuntimeError as e:
            print(f"错误: {e}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)

    if args.status:
        state = tm.load()
        if state is None:
            print("[business_platform] 无本地 token 状态，请先登录")
        else:
            remain = tm.expires_in_seconds(state)
            print(f"  username:     {state.username}")
            print(f"  user_id:      {state.user_id}")
            print(f"  tenant_id:    {state.tenant_id}")
            print(f"  expires_in:   {int(remain)}s（约 {int(remain // 60)} 分钟）")
            print(f"  expired:      {tm.is_expired(state)}")
            print(f"  access_token: {state.access_token[:50]}...")
        sys.exit(0)

    # 默认行为：登录
    tm.login()
