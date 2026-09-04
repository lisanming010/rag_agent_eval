"""
业务平台 API 调用客户端

主线流程：
    1. 从 config.yaml 的 business_platform 节读取登录信息（login_url / username / password）
    2. 登录业务平台获取 access_token（复用 BusinessPlatformTokenManager，支持本地缓存与过期自动重登）
    3. 调用业务接口采集实体数据，导出为实体映射（entity_mapping.json）：
       - 电站 / 项目：/api/power-station/overview/list-v2，提取 id / name / settlementOrganizationName
       - 设备：/api/device/basic-list-page，按类型（Inverter / MeteorologicalStation）每类各取前
         device_page_size 条，提取 id / name / siteId / deviceSn

典型用法：
    from tool.business_platform_client import BusinessPlatformClient

    client = BusinessPlatformClient()   # 自动从 config.yaml 读取登录配置
    client.main_flow()                  # 主线流程：登录 → 调用置顶接口获取指定值
    client.export_entity_mapping()      # 登录 → 调接口 → 导出实体映射到 test_suite/dataqa/entity_mapping.json

命令行用法：
    python -m tool.business_platform_client                 # 走完整主线流程
    python -m tool.business_platform_client --export-mapping  # 导出实体映射
    python -m tool.business_platform_client --login-only     # 仅登录，验证登录环节
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from tool.business_platform_token_manager import (
    BusinessPlatformTokenManager,
    PlatformTokenState,
)
from tool.config_reader import ConfigReader

# 实体映射文件默认路径（相对项目根目录）
DEFAULT_ENTITY_MAPPING_PATH = (
    Path(__file__).resolve().parent.parent / "test_suite" / "dataqa" / "entity_mapping.json"
)

# 设备信息采集默认值（device_page_size 可从 config.yaml 的 business_platform.device_page_size 覆盖）
DEFAULT_DEVICE_PAGE_SIZE = int(ConfigReader.get_instance().get("business_platform.device_page_size", 10))
# 设备类型枚举：Inverter（逆变器）/ MeteorologicalStation（气象站），每类各取前 device_page_size 条
DEFAULT_DEVICE_TYPES = ["Inverter", "MeteorologicalStation"]
# 采集字段
DEFAULT_DEVICE_FIELDS = ["id", "name", "siteId", "deviceSn"]


class BusinessPlatformClient:
    """业务平台 API 调用客户端

    职责：
        1. 从 config.yaml 读取业务平台登录配置
        2. 登录 / 复用有效 token（鉴权）
        3. 以鉴权状态调用业务平台 API
        4. 主线流程：登录后调用置顶接口获取指定值（待实施）
    """

    def __init__(
        self,
        login_url: str | None = None,
        username: str | None = None,
        password: str | None = None,
        api_base_url: str | None = None,
        device_page_size: int | None = None,
    ) -> None:
        """参数缺省时从 config.yaml 的 business_platform 节读取"""
        conf = ConfigReader.get_instance()
        self.api_base_url = api_base_url or conf.get("business_platform.api_base_url", "") or ""
        self.device_page_size = device_page_size or DEFAULT_DEVICE_PAGE_SIZE
        self.device_types = list(DEFAULT_DEVICE_TYPES)
        self.device_fields = list(DEFAULT_DEVICE_FIELDS)
        self._token_manager = BusinessPlatformTokenManager(
            login_url=login_url or conf.get("business_platform.login_url", ""),
            username=username or conf.get("business_platform.username", ""),
            password=password or conf.get("business_platform.password", ""),
        )

    # ---------- 登录 / 鉴权 ----------
    def login(self) -> PlatformTokenState:
        """登录业务平台并持久化 token"""
        return self._token_manager.login()

    def ensure_login(self) -> str:
        """确保已登录，返回有效 access_token（本地缓存有效则复用，过期自动重登）"""
        return self._token_manager.get_valid_token()

    def auth_headers(self) -> dict[str, str]:
        """获取带 Authorization 的请求头（自动确保 token 有效）"""
        return self._token_manager.get_auth_header()

    # ---------- 通用 API 调用 ----------
    def call_api(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        json_body: dict | None = None,
        timeout: int = 30,
    ) -> Any:
        """带鉴权调用业务平台 API

        Args:
            method: HTTP 方法（GET / POST / PUT / DELETE 等）
            endpoint: 接口路径，如 "/api/xxx"（相对 api_base_url）
            params: URL 查询参数
            json_body: JSON 请求体
            timeout: 请求超时秒数

        Returns:
            Any: 响应的 data 字段；若响应无 data 字段则返回整个 body

        Raises:
            RuntimeError: 未配置 api_base_url、请求失败或业务返回错误
        """
        if not self.api_base_url:
            raise RuntimeError("未配置 business_platform.api_base_url，无法调用业务平台接口")

        url = self.api_base_url.rstrip("/") + "/" + endpoint.lstrip("/")
        try:
            resp = requests.request(
                method,
                url,
                params=params,
                json=json_body,
                headers=self.auth_headers(),
                timeout=timeout,
                verify=True,
            )
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as e:
            raise RuntimeError(f"业务平台接口调用失败: {e}") from e
        except ValueError as e:
            raise RuntimeError(f"业务平台接口响应解析失败: {e}") from e

        code = body.get("code")
        if code is not None and code != 200:
            msg = body.get("msg", "未知错误")
            raise RuntimeError(f"业务平台接口返回错误: [{code}] {msg}")

        return body.get("data", body)

    # ---------- 置顶接口：电站信息查询 ----------
    def fetch_power_stations(self, page_num: int = 1, page_size: int = 15000) -> list[dict]:
        """调用置顶接口 /api/power-station/overview/list-v2 查询电站列表

        POST 请求体: {"pageNum": 1, "pageSize": 15000}
        响应结构（rows 在顶层，无 data 包裹）:
            {"total": "7", "rows": [{id, name, settlementOrganizationName, ...}], "code": 200, "msg": "查询成功"}

        Args:
            page_num: 页码
            page_size: 每页条数

        Returns:
            list[dict]: 电站 rows 列表（每项含 id / name / settlementOrganizationName 等字段）
        """
        data = self.call_api(
            "POST",
            "/api/power-station/overview/list-v2",
            json_body={"pageNum": page_num, "pageSize": page_size},
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def get_pinned_value(self) -> list[dict]:
        """调用置顶接口获取指定值（电站列表）

        Returns:
            list[dict]: 电站 rows 列表
        """
        return self.fetch_power_stations()

    # ---------- 设备信息采集 ----------
    def fetch_devices(self, device_type: str, page_size: int | None = None) -> list[dict]:
        """调用设备信息接口 /api/device/basic-list-page 获取指定类型的设备列表

        GET 参数: name / pageNum / pageSize / params[name_like] / keyword / orderByColumn /
                  isAsc / deviceType / usageStatus / _t
        其中 _t 传递当前登录用户的租户 ID（从 token 状态中获取）。

        Args:
            device_type: 设备类型枚举（Inverter 逆变器 / MeteorologicalStation 气象站）
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 设备 rows 列表（含 id / name / siteId / deviceSn 等字段）
        """
        page_size = page_size or self.device_page_size
        tenant_id = ""
        state = self._token_manager.get_state()
        if state:
            tenant_id = state.tenant_id
        params = {
            "name": "",
            "pageNum": 1,
            "pageSize": page_size,
            "params[name_like]": "",
            "keyword": "",
            "orderByColumn": "id",
            "isAsc": "descend",
            "deviceType": device_type,
            "usageStatus": 1,
            "_t": tenant_id,
        }
        data = self.call_api("GET", "/api/device/basic-list-page", params=params)
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def fetch_all_devices(self) -> dict[str, list[dict]]:
        """按设备类型枚举，每类各采集前 device_page_size 条设备

        Returns:
            dict: {deviceType: [设备 rows, ...], ...}
        """
        result: dict[str, list[dict]] = {}
        for device_type in self.device_types:
            rows = self.fetch_devices(device_type)
            result[device_type] = rows
            print(f"[business_platform_client] {device_type}: 采集到 {len(rows)} 条设备")
        return result

    def build_device_mapping(self, devices: dict[str, list[dict]]) -> dict:
        """从设备 rows 中提取采集字段（id / name / siteId / deviceSn），构建实体映射

        Args:
            devices: fetch_all_devices() 返回的 {deviceType: rows}

        Returns:
            dict: {deviceType: [{field: value, ...}, ...]}，每条仅保留采集字段
        """
        mapping: dict[str, list[dict]] = {}
        for device_type, rows in devices.items():
            items = []
            for row in rows:
                item = {field: row.get(field) for field in self.device_fields}
                # id 统一为字符串
                if item.get("id") is not None:
                    item["id"] = str(item["id"])
                items.append(item)
            mapping[device_type] = items
        return mapping

    # ---------- 结算组织（项目）采集 ----------
    def fetch_settlement_organizations(self, page_size: int | None = None) -> list[dict]:
        """调用结算组织接口 /api/system/settlement-organization/list 获取项目列表

        GET 参数: orderByColumn=id / isAsc=descend / pageNum / pageSize / _t
        其中 _t 传递当前登录用户的租户 ID。
        响应结构（rows 在顶层）:
            {"total": "2", "rows": [{id, name, ...}], "code": 200, "msg": "..."}
        row.name 即电站接口中的 settlementOrganizationName。

        Args:
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 结算组织 rows 列表（含 id / name 等字段）
        """
        page_size = page_size or self.device_page_size
        tenant_id = ""
        state = self._token_manager.get_state()
        if state:
            tenant_id = state.tenant_id
        params = {
            "orderByColumn": "id",
            "isAsc": "descend",
            "pageNum": 1,
            "pageSize": page_size,
            "_t": tenant_id,
        }
        data = self.call_api("GET", "/api/system/settlement-organization/list", params=params)
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def fill_project_ids(self, mapping: dict, orgs: list[dict]) -> dict:
        """根据结算组织接口返回的项目列表，回填"项目"条目的 id

        以名称匹配关联：org.name 对应映射中项目条目的 name。
        项目 id 为雪花 id，超过 JS 安全整数范围，保留字符串格式避免精度丢失。

        Args:
            mapping: build_entity_mapping() 生成的映射（含"项目"分类）
            orgs: fetch_settlement_organizations() 返回的项目列表

        Returns:
            dict: 回填 id 后的映射
        """
        org_id_by_name: dict[str, str] = {}
        for org in orgs:
            org_id = org.get("id")
            org_name = org.get("name") or org.get("settlementOrganizationName")
            if org_id is not None and org_name:
                org_id_by_name[org_name] = str(org_id)
        filled = 0
        for proj in mapping.get("项目", []):
            if not proj.get("id") and proj["name"] in org_id_by_name:
                proj["id"] = org_id_by_name[proj["name"]]
                filled += 1
        print(f"[business_platform_client] 项目 id 回填: {filled}/{len(mapping.get('项目', []))}")
        return mapping

    # ---------- 消缺工单采集 ----------
    def fetch_work_orders(self, page_size: int | None = None) -> list[dict]:
        """调用消缺工单接口 /api/work-order/list-v2 获取工单列表

        POST 请求体: {"pageNum": 1, "pageSize": N}
        响应结构（rows 在顶层）:
            {"total": "...", "rows": [{id, workOrderStatus, level, chargePersonId, chargePerson, ...}], ...}

        Args:
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 工单 rows 列表
        """
        page_size = page_size or self.device_page_size
        data = self.call_api(
            "POST",
            "/api/work-order/list-v2",
            json_body={"pageNum": 1, "pageSize": page_size},
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def build_work_order_mapping(self, rows: list[dict]) -> list[dict]:
        """从工单 rows 中提取采集字段，构建"消缺工单"映射条目

        采集字段: id / workOrderStatus / level / chargePersonId / chargePerson
        id 统一为字符串。

        Args:
            rows: fetch_work_orders() 返回的工单列表

        Returns:
            list[dict]: 工单条目列表
        """
        fields = ["id", "workOrderStatus", "level", "chargePersonId", "chargePerson",
                  "powerStationNames", "powerStationIds"]
        items: list[dict] = []
        for row in rows:
            item = {f: row.get(f) for f in fields}
            if item.get("id") is not None:
                item["id"] = str(item["id"])
            items.append(item)
        return items

    # ---------- 消缺记录采集 ----------
    def fetch_work_order_records(self, page_size: int | None = None) -> list[dict]:
        """调用消缺工单接口查询已完成的消缺记录

        POST body: {"orderByColumn":"startTime","isAsc":"descend","pageNum":1,
                     "pageSize":N,"workOrderStatus":["Complete","Cancel","BeEvaluated","AcceptancePassed"]}
        即筛选已完成/已取消/待评价/已验收通过的工单。

        Args:
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 消缺记录 rows 列表
        """
        page_size = page_size or self.device_page_size
        data = self.call_api(
            "POST",
            "/api/work-order/list-v2",
            json_body={
                "orderByColumn": "startTime",
                "isAsc": "descend",
                "pageNum": 1,
                "pageSize": page_size,
                "workOrderStatus": ["Complete", "Cancel", "BeEvaluated", "AcceptancePassed"],
            },
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def build_work_order_record_mapping(self, rows: list[dict]) -> list[dict]:
        """从消缺记录 rows 中提取采集字段，构建"消缺记录"映射条目

        采集字段: id / workOrderStatus / level / chargePersonId / chargePerson /
                  powerStationNames / powerStationIds
        id 统一为字符串。

        Args:
            rows: fetch_work_order_records() 返回的消缺记录列表

        Returns:
            list[dict]: 消缺记录条目列表
        """
        fields = [
            "id", "workOrderStatus", "level", "chargePersonId", "chargePerson",
            "powerStationNames", "powerStationIds",
        ]
        items: list[dict] = []
        for row in rows:
            item = {f: row.get(f) for f in fields}
            if item.get("id") is not None:
                item["id"] = str(item["id"])
            items.append(item)
        return items

    # ---------- 低效告警白名单采集 ----------
    def fetch_warning_whitelist(self, page_size: int | None = None) -> list[dict]:
        """调用低效告警白名单接口 /api/power-generation-warning-whitelist/list

        POST body: {"orderByColumn":"id","isAsc":"descend","pageNum":1,"pageSize":N}
        提取值: id / changedUserId / changedBy / stationId / deviceId

        Args:
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 白名单 rows 列表
        """
        page_size = page_size or self.device_page_size
        data = self.call_api(
            "POST",
            "/api/power-generation-warning-whitelist/list",
            json_body={"orderByColumn": "id", "isAsc": "descend",
                       "pageNum": 1, "pageSize": page_size},
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def build_warning_whitelist_mapping(self, rows: list[dict]) -> list[dict]:
        """从白名单 rows 中提取采集字段，构建"低效告警白名单"映射条目

        采集字段: id / changedUserId / changedBy / stationId / deviceId
        id 统一为字符串。

        Args:
            rows: fetch_warning_whitelist() 返回的白名单列表

        Returns:
            list[dict]: 白名单条目列表
        """
        fields = ["id", "changedUserId", "changedBy", "stationId", "deviceId"]
        items: list[dict] = []
        for row in rows:
            item = {f: row.get(f) for f in fields}
            if item.get("id") is not None:
                item["id"] = str(item["id"])
            items.append(item)
        return items

    # ---------- 隐患工单采集 ----------
    def fetch_hidden_flaw_records(self, page_size: int | None = None) -> list[dict]:
        """调用隐患工单接口 /api/flaw/hidden-flaw-record/list-v2

        POST body: {"orderByColumn":"id","isAsc":"descend","pageNum":1,"pageSize":N}

        Args:
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 隐患工单 rows 列表
        """
        page_size = page_size or self.device_page_size
        data = self.call_api(
            "POST",
            "/api/flaw/hidden-flaw-record/list-v2",
            json_body={"orderByColumn": "id", "isAsc": "descend",
                       "pageNum": 1, "pageSize": page_size},
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def build_hidden_flaw_record_mapping(self, rows: list[dict]) -> list[dict]:
        """从隐患工单 rows 中提取采集字段，构建"隐患工单"映射条目

        采集字段: id / powerStationId / powerStationName / rectifyPersonId /
                  rectifyPersonName
        id 统一为字符串。

        Args:
            rows: fetch_hidden_flaw_records() 返回的隐患工单列表

        Returns:
            list[dict]: 隐患工单条目列表
        """
        fields = [
            "id", "powerStationId", "powerStationName", "rectifyPersonId",
            "rectifyPersonName",
        ]
        items: list[dict] = []
        for row in rows:
            item = {f: row.get(f) for f in fields}
            if item.get("id") is not None:
                item["id"] = str(item["id"])
            items.append(item)
        return items

    # ---------- 巡检工单采集 ----------
    def fetch_inspection_work_orders(self, page_size: int | None = None) -> list[dict]:
        """调用巡检工单接口 /api/event/inspection-work-order/list-v2

        POST body: {"orderByColumn":"createdDate","isAsc":"descend","pageNum":1,"pageSize":N}

        Args:
            page_size: 每页条数，缺省使用 config 配置的 device_page_size

        Returns:
            list[dict]: 巡检工单 rows 列表
        """
        page_size = page_size or self.device_page_size
        data = self.call_api(
            "POST",
            "/api/event/inspection-work-order/list-v2",
            json_body={"orderByColumn": "createdDate", "isAsc": "descend",
                       "pageNum": 1, "pageSize": page_size},
        )
        rows = data.get("rows", []) if isinstance(data, dict) else []
        return rows

    def build_inspection_work_order_mapping(self, rows: list[dict]) -> list[dict]:
        """从巡检工单 rows 中提取采集字段，构建"巡检工单"映射条目

        采集字段: id / powerStationId / powerStationName / chargePersonId /
                  chargePersonName / status
        id 统一为字符串。

        Args:
            rows: fetch_inspection_work_orders() 返回的巡检工单列表

        Returns:
            list[dict]: 巡检工单条目列表
        """
        fields = [
            "id", "powerStationId", "powerStationName", "chargePersonId",
            "chargePersonName", "status",
        ]
        items: list[dict] = []
        for row in rows:
            item = {f: row.get(f) for f in fields}
            if item.get("id") is not None:
                item["id"] = str(item["id"])
            items.append(item)
        return items

    # ---------- 实体映射导出 ----------
    def build_entity_mapping(self, rows: list[dict]) -> dict:
        """从电站 rows 中提取 id / name / settlementOrganizationName，构建实体映射

        映射格式与 test_suite/dataqa/entity_mapping.json 保持一致（id 统一为字符串）：
            "电站": [{"name": 电站名称, "id": 电站 id, "projectName": 所属项目名称,
                      "provinceCityDistrict": 省市区}, ...]
            "项目": [{"name": 结算组织名称（去重）, "id": 项目 id, "stations": [电站名称, ...]}, ...]

        说明：接口未提供结算组织（项目）独立的 id，项目条目 id 暂留空字符串，
        待后续接口补充后再回填；项目条目通过 "stations" 字段保留与电站的关联。

        Args:
            rows: fetch_power_stations() 返回的电站列表

        Returns:
            dict: 仅含 电站 / 项目 两个分类的实体映射
        """
        stations: list[dict] = []
        projects: dict[str, dict] = {}
        for row in rows:
            station_id = row.get("id")
            name = row.get("name")
            org = row.get("settlementOrganizationName")
            if station_id is not None and name:
                stations.append({
                    "name": name,
                    "id": str(station_id),
                    "projectName": org or "",
                    "provinceCityDistrict": row.get("provinceCityDistrict") or "",
                })
            if org:
                project = projects.setdefault(org, {"name": org, "id": "", "stations": []})
                if name:
                    project["stations"].append(name)
        return {"电站": stations, "项目": list(projects.values())}

    def export_entity_mapping(self, output_path: str | Path | None = None) -> Path:
        """调用业务平台接口并导出实体映射到 entity_mapping.json

        采集内容：
            - 电站 / 项目：置顶接口 /api/power-station/overview/list-v2
            - 设备：/api/device/basic-list-page（按 device_types 每类各取前 device_page_size 条）

        文件已存在时保留其他分类（区域 / 地址等手工数据），仅更新采集到的分类。

        Args:
            output_path: 输出文件路径，缺省为 test_suite/dataqa/entity_mapping.json

        Returns:
            Path: 写入的文件路径
        """
        path = Path(output_path) if output_path else DEFAULT_ENTITY_MAPPING_PATH

        # 电站 / 项目
        rows = self.fetch_power_stations()
        mapping = self.build_entity_mapping(rows)

        # 项目 id 回填（结算组织接口）
        orgs = self.fetch_settlement_organizations()
        mapping = self.fill_project_ids(mapping, orgs)

        # 设备（逆变器 / 气象站）
        devices = self.fetch_all_devices()
        mapping.update(self.build_device_mapping(devices))

        # 消缺工单
        orders = self.fetch_work_orders()
        mapping["消缺工单"] = self.build_work_order_mapping(orders)

        # 消缺记录（已完成的工单）
        records = self.fetch_work_order_records()
        mapping["消缺记录"] = self.build_work_order_record_mapping(records)

        # 巡检工单
        inspections = self.fetch_inspection_work_orders()
        mapping["巡检工单"] = self.build_inspection_work_order_mapping(inspections)

        # 隐患工单
        flaws = self.fetch_hidden_flaw_records()
        mapping["隐患工单"] = self.build_hidden_flaw_record_mapping(flaws)

        # 低效告警白名单
        whitelist = self.fetch_warning_whitelist()
        mapping["低效告警白名单"] = self.build_warning_whitelist_mapping(whitelist)

        existing: dict = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
        existing.update(mapping)

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(existing, f, ensure_ascii=False, indent=2)
        print(
            f"[business_platform_client] 实体映射已写入: {path} "
            f"（电站 {len(mapping['电站'])} 条，项目 {len(mapping['项目'])} 条，"
            f"设备 {sum(len(v) for v in devices.values())} 条，工单 {len(mapping['消缺工单'])} 条，"
            f"消缺记录 {len(mapping['消缺记录'])} 条，"
            f"巡检工单 {len(mapping['巡检工单'])} 条，"
            f"隐患工单 {len(mapping['隐患工单'])} 条，"
            f"白名单 {len(mapping['低效告警白名单'])} 条）"
        )
        return path

    # ---------- 主线流程 ----------
    def main_flow(self) -> list[dict]:
        """主线流程：登录业务平台 → 调用置顶接口获取指定值

        Returns:
            list[dict]: 置顶接口返回的电站列表
        """
        self.ensure_login()
        rows = self.fetch_power_stations()
        print(f"[business_platform_client] 登录成功 (user: {self._token_manager.username})")
        print(f"[business_platform_client] 置顶接口返回 {len(rows)} 条电站数据")
        return rows


# ---------- 命令行入口 ----------
if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="业务平台 API 调用客户端")
    parser.add_argument("--export-mapping", nargs="?", const="", default=None,
                        help="调用置顶接口并导出实体映射（可指定输出路径，缺省为 entity_mapping.json）")
    parser.add_argument("--login-only", action="store_true", help="仅登录验证，不调用业务接口")
    args = parser.parse_args()

    client = BusinessPlatformClient()

    if args.login_only:
        state = client.login()
        print(f"[business_platform_client] 登录成功 (user: {state.username}, tenant: {state.tenant_id})")
        sys.exit(0)

    if args.export_mapping is not None:
        client.export_entity_mapping(args.export_mapping or None)
        sys.exit(0)

    rows = client.main_flow()
    print(f"[business_platform_client] 置顶接口获取到 {len(rows)} 条电站数据")
