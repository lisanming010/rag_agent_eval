"""
Playwright 手动登录工具

使用场景：
    目标站点需要人工登录，登录后将浏览器中的 cookies / localStorage / sessionStorage
    持久化到本地，供后续 API 调用复用；同时记录鉴权过期时间点，供调用方判断
    是否需要重新触发登录。

典型用法：
    from tool.playwright_login import PlaywrightLogin

    login = PlaywrightLogin()                   # 使用默认站点与状态文件
    if login.is_expired():                      # 接口调用前先判断
        login.login()                           # 弹出浏览器让用户手动登录
    cookies = login.get_cookie_header()         # 取到鉴权头供 requests/aiohttp 使用

命令行用法：
    python -m tool.playwright_login             # 触发一次手动登录并保存状态
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

DEFAULT_LOGIN_URL = "http://aiforge-serving.tecdo.cn:8041/login"
# 保存到项目根目录而非 tool/ 子目录
DEFAULT_STATE_FILE = Path(__file__).resolve().parent.parent / "auth_state.json"

# Venus AdPilot 使用 OAuth2 + JWT，access_token 无明确过期时间，使用回退时长（秒）
# 实际过期由后端 /api/v1/auth/refresh 接口 401 判定
DEFAULT_FALLBACK_TTL_SECONDS = 12 * 60 * 60

# 判断过期时预留的安全余量（秒），避免卡在临界点失败
EXPIRY_SAFETY_MARGIN_SECONDS = 60

# Venus AdPilot 登录成功标志：localStorage 中存在 access_token
VENUS_ACCESS_TOKEN_KEY = "access_token"


@dataclass
class AuthState:
    """本地持久化的鉴权状态"""
    login_url: str
    origin: str
    saved_at: float
    expires_at: float
    cookies: list[dict[str, Any]] = field(default_factory=list)
    local_storage: dict[str, str] = field(default_factory=dict)
    session_storage: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "login_url": self.login_url,
            "origin": self.origin,
            "saved_at": self.saved_at,
            "expires_at": self.expires_at,
            "cookies": self.cookies,
            "local_storage": self.local_storage,
            "session_storage": self.session_storage,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuthState":
        return cls(
            login_url=data["login_url"],
            origin=data["origin"],
            saved_at=float(data["saved_at"]),
            expires_at=float(data["expires_at"]),
            cookies=data.get("cookies", []),
            local_storage=data.get("local_storage", {}),
            session_storage=data.get("session_storage", {}),
        )


class PlaywrightLogin:
    def __init__(
        self,
        login_url: str = DEFAULT_LOGIN_URL,
        state_file: str | Path = DEFAULT_STATE_FILE,
        fallback_ttl_seconds: int = DEFAULT_FALLBACK_TTL_SECONDS,
        wait_timeout_seconds: int = 300,
    ) -> None:
        self.login_url = login_url
        self.state_file = Path(state_file)
        self.fallback_ttl_seconds = fallback_ttl_seconds
        self.wait_timeout_seconds = wait_timeout_seconds

        parsed = urlparse(login_url)
        self.origin = f"{parsed.scheme}://{parsed.netloc}"

    # ---------- 状态文件管理 ----------
    def load(self) -> AuthState | None:
        if not self.state_file.exists():
            return None
        try:
            with self.state_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return AuthState.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def _save(self, state: AuthState) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(self.state_file.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.state_file)

    def clear(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()

    # ---------- 过期判断 ----------
    def is_expired(self, state: AuthState | None = None) -> bool:
        state = state or self.load()
        if state is None:
            return True
        return time.time() >= (state.expires_at - EXPIRY_SAFETY_MARGIN_SECONDS)

    def expires_in_seconds(self, state: AuthState | None = None) -> float:
        state = state or self.load()
        if state is None:
            return 0.0
        return max(0.0, state.expires_at - time.time())

    # ---------- 便捷的取鉴权信息接口 ----------
    def get_cookies_dict(self, state: AuthState | None = None) -> dict[str, str]:
        state = state or self.load()
        if state is None:
            return {}
        return {c["name"]: c["value"] for c in state.cookies}

    def get_cookie_header(self, state: AuthState | None = None) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.get_cookies_dict(state).items())

    def get_token(self, keys: tuple[str, ...] = ("token", "access_token", "Authorization")) -> str | None:
        """从 localStorage / sessionStorage / cookie 中顺序尝试取出常见 token 字段"""
        state = self.load()
        if state is None:
            return None
        for store in (state.local_storage, state.session_storage):
            for k in keys:
                if k in store and store[k]:
                    return store[k]
        cookies = {c["name"]: c["value"] for c in state.cookies}
        for k in keys:
            if k in cookies and cookies[k]:
                return cookies[k]
        return None

    # ---------- 核心：启动浏览器让用户手动登录 ----------
    def login(self, headless: bool = False) -> AuthState:
        """
        启动浏览器导航到登录页，等待用户登录成功后捕获鉴权信息并保存。

        Venus AdPilot 登录流程：
            1) 访问 /login 页面
            2) 点击登录按钮跳转到 CAS OAuth2 授权页面
            3) 在 CAS 完成登录后回调到 /auth/callback
            4) 前端用 code 换取 access_token 并存入 localStorage

        登录完成判定：localStorage 中存在 access_token
        """
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "未安装 playwright。请先执行：pipenv install playwright 且 pipenv run playwright install chromium"
            ) from e

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            page.goto(self.login_url)

            print(f"[playwright_login] 已打开登录页: {self.login_url}")
            print("[playwright_login] 请在弹出的浏览器中完成以下步骤：")
            print("    1. 点击登录按钮（会跳转到 CAS 授权页面）")
            print("    2. 在 CAS 页面输入账号密码完成登录")
            print("    3. 登录成功后会自动回调，等待页面加载完成")
            print("    4. 脚本会自动检测 localStorage 中的 access_token")
            print()
            print("[playwright_login] 正在等待登录完成...")

            deadline = time.time() + self.wait_timeout_seconds
            access_token = None

            while time.time() < deadline:
                try:
                    # 检查 localStorage 中是否有 access_token
                    local_storage = page.evaluate(
                        "() => Object.fromEntries(Object.entries(window.localStorage))"
                    )
                    access_token = local_storage.get(VENUS_ACCESS_TOKEN_KEY)
                    if access_token:
                        print(f"[playwright_login] ✓ 检测到 access_token，登录成功！")
                        break
                except Exception:
                    pass

                try:
                    page.wait_for_timeout(1000)
                except Exception:
                    break

            if not access_token:
                print("[playwright_login] ⚠ 未检测到 access_token，尝试手动确认...")
                try:
                    input("[playwright_login] 如果已完成登录，请按回车继续...")
                    local_storage = page.evaluate(
                        "() => Object.fromEntries(Object.entries(window.localStorage))"
                    )
                    access_token = local_storage.get(VENUS_ACCESS_TOKEN_KEY)
                except (EOFError, Exception):
                    local_storage = {}

            # 获取所有鉴权信息
            cookies = context.cookies()
            try:
                local_storage = page.evaluate(
                    "() => Object.fromEntries(Object.entries(window.localStorage))"
                )
            except Exception:
                local_storage = {}
            try:
                session_storage = page.evaluate(
                    "() => Object.fromEntries(Object.entries(window.sessionStorage))"
                )
            except Exception:
                session_storage = {}

            context.close()
            browser.close()

        # 验证是否成功获取 access_token
        if not local_storage.get(VENUS_ACCESS_TOKEN_KEY):
            raise RuntimeError(
                f"登录失败：未能从 localStorage 中获取 {VENUS_ACCESS_TOKEN_KEY}。"
                "请确保已完成登录流程并等待页面加载完成。"
            )

        expires_at = self._compute_expires_at(cookies, local_storage)
        state = AuthState(
            login_url=self.login_url,
            origin=self.origin,
            saved_at=time.time(),
            expires_at=expires_at,
            cookies=cookies,
            local_storage=local_storage or {},
            session_storage=session_storage or {},
        )
        self._save(state)

        remain = max(0, int(expires_at - time.time()))
        print(
            f"[playwright_login] 鉴权已保存至 {self.state_file}，"
            f"预计剩余有效期约 {remain} 秒（{remain // 60} 分钟）。"
        )
        print(f"[playwright_login] access_token: {local_storage.get(VENUS_ACCESS_TOKEN_KEY)[:20]}...")
        return state

    # ---------- 内部：根据 cookie 推导过期时间 ----------
    def _compute_expires_at(self, cookies: list[dict[str, Any]], local_storage: dict[str, str] | None = None) -> float:
        """
        计算鉴权过期时间。

        Venus AdPilot 使用 JWT access_token，优先从目标域的 refresh_token / access_token cookie 推导。

        Args:
            cookies: 浏览器 cookies
            local_storage: localStorage 内容（保留参数以便未来扩展 JWT 解析）
        """
        now = time.time()
        target_domain = urlparse(self.origin).netloc

        # 优先查找目标域的 refresh_token / access_token cookie
        auth_cookies: list[float] = []
        for c in cookies:
            if c.get("domain", "").endswith(target_domain) and c.get("name") in ("refresh_token", "access_token"):
                exp = c.get("expires")
                if isinstance(exp, (int, float)) and exp > 0 and exp > now:
                    auth_cookies.append(float(exp))

        if auth_cookies:
            # 使用 access_token 的过期时间（通常比 refresh_token 短）
            return min(auth_cookies)

        # 回退：从所有目标域的持久化 cookie 中推导（排除第三方域如 GA、飞书等）
        domain_cookies: list[float] = []
        for c in cookies:
            domain = c.get("domain", "")
            if domain.endswith(target_domain):
                exp = c.get("expires")
                if isinstance(exp, (int, float)) and exp > 0 and exp > now:
                    domain_cookies.append(float(exp))

        if domain_cookies:
            return min(domain_cookies)

        return now + self.fallback_ttl_seconds


def ensure_logged_in(
    login_url: str = DEFAULT_LOGIN_URL,
    state_file: str | Path = DEFAULT_STATE_FILE,
) -> AuthState:
    """便捷函数：若当前状态已过期则触发手动登录，返回可用的鉴权状态。"""
    helper = PlaywrightLogin(login_url=login_url, state_file=state_file)
    if helper.is_expired():
        return helper.login()
    state = helper.load()
    assert state is not None
    return state


if __name__ == "__main__":
    helper = PlaywrightLogin()
    state = helper.login()
    print("cookie header:", helper.get_cookie_header(state))
    print("expires_at:", state.expires_at, "remain:", helper.expires_in_seconds(state))
