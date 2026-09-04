"""占位符填充器：在执行前将测试数据集中的占位符替换为实际值。

两类占位符（均需以 `[xxx]` 反引号包裹书写；裸 [xxx] 是普通文本，不参与替换）：
  1. 实体类    — 从 entity_mapping.json 按分类随机抽取实体（支持 [电站] 锚定 + 关联抽取）
  2. 时间类    — 基于执行当天动态计算（固定写法 [今天起始] 等；泛化写法
     [N天前起始/结束]、[N个月前起始/结束]、[N年前起始/结束]，N 为数字）

detailList 等监控项参数值（如 [直流电压1,当日发电量]）是普通文本，裸写原样保留。

反引号包裹但未识别的占位符 fail fast 报错（携带文件、行号、列名），避免残留
字面 token 悄悄进入 Agent 调用导致难以排查的失败。
"""

import calendar
import json
import os
import random
import re
from datetime import datetime, timedelta
from pathlib import Path

from tool.config_reader import ConfigReader
from tool.log_factory import LogFactory

logger = LogFactory.get_logger(__name__)

# 占位符标识为反引号包裹的 `[xxx]`，替换时反引号一并吞掉；
# 裸 [xxx] 是普通文本，不参与匹配与替换（如 detailList=[直流电压1,当日发电量]）
TOKEN_RE = re.compile(r'`\[[^\]]+\]`')

_DT_FMT = '%Y-%m-%d %H:%M:%S'

# ---------------------------------------------------------------------------
# 时间占位符（动态计算，基准 = 执行时刻）
# ---------------------------------------------------------------------------

_TIME_TOKENS = frozenset({
    '[今天起始]', '[今天结束]', '[本月起始]', '[本月结束]', '[今年起始]', '[今年结束]',
    '[上月起始]', '[上月结束]', '[去年起始]', '[去年结束]',
    '[3天前起始]', '[7天前起始]', '[一周前起始]', '[24小时前]', '[当前时刻]',
})

# 泛化相对时间占位符: [N天前起始/结束]、[N个月前起始/结束]、[N年前起始/结束]
# 单位必须是"个月"（[3月前起始] 不匹配，避免与月份编号混淆），N 为数字（含 0）
_REL_TIME_RE = re.compile(r'\[(\d+)(天|个月|年)前(起始|结束)\]')


def _time_value(token: str, now: datetime) -> str:
    """按 token 计算时间值，返回 '%Y-%m-%d %H:%M:%S' 字符串"""
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if token == '[今天起始]':
        return today.strftime(_DT_FMT)
    if token == '[今天结束]':
        return (today + timedelta(days=1) - timedelta(seconds=1)).strftime(_DT_FMT)
    if token == '[本月起始]':
        return today.replace(day=1).strftime(_DT_FMT)
    if token == '[本月结束]':
        # 下月1日的前一秒
        next_month_first = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
        return (next_month_first - timedelta(seconds=1)).strftime(_DT_FMT)
    if token == '[今年起始]':
        return today.replace(month=1, day=1).strftime(_DT_FMT)
    if token == '[今年结束]':
        return today.replace(month=12, day=31).strftime(_DT_FMT)
    if token == '[上月起始]':
        last_month_last_day = today.replace(day=1) - timedelta(days=1)
        return last_month_last_day.replace(day=1).strftime(_DT_FMT)
    if token == '[上月结束]':
        return (today.replace(day=1) - timedelta(seconds=1)).strftime(_DT_FMT)
    if token == '[去年起始]':
        return today.replace(year=today.year - 1, month=1, day=1).strftime(_DT_FMT)
    if token == '[去年结束]':
        return today.replace(year=today.year - 1, month=12, day=31).strftime(_DT_FMT)
    if token == '[3天前起始]':
        return (today - timedelta(days=3)).strftime(_DT_FMT)
    if token == '[7天前起始]' or token == '[一周前起始]':
        return (today - timedelta(days=7)).strftime(_DT_FMT)
    if token == '[24小时前]':
        return (now - timedelta(hours=24)).strftime(_DT_FMT)
    if token == '[当前时刻]':
        return now.strftime(_DT_FMT)
    raise ValueError(f'未知时间占位符: {token}')


def _shift_months(d: datetime, n: int) -> datetime:
    """日历语义向前平移 n 个月，日号钳制到目标月最后一天（如 3-31 减 1 个月 → 2-28/29）"""
    idx = d.month - 1 - n
    year = d.year + idx // 12
    month = idx % 12 + 1
    day = min(d.day, calendar.monthrange(year, month)[1])
    return d.replace(year=year, month=month, day=day)


def _shift_years(d: datetime, n: int) -> datetime:
    """向前平移 n 年，闰日（2-29）钳制到 2-28"""
    year = d.year - n
    day = min(d.day, calendar.monthrange(year, d.month)[1])
    return d.replace(year=year, month=d.month, day=day)


def _rel_time_value(match: re.Match, now: datetime) -> str:
    """按泛化相对时间占位符 [N天/月/年前起始/结束] 计算时间值。

    天: 基准日直接平移 N 天；月/年: 日历语义平移（月末/闰日钳制）。
    起始 → 当天 00:00:00，结束 → 当天 23:59:59。
    """
    n = int(match.group(1))
    unit = match.group(2)
    boundary = match.group(3)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    if unit == '天':
        base = today - timedelta(days=n)
    elif unit == '个月':
        base = _shift_months(today, n)
    else:  # '年'
        base = _shift_years(today, n)

    if boundary == '结束':
        base += timedelta(days=1) - timedelta(seconds=1)
    return base.strftime(_DT_FMT)


# ---------------------------------------------------------------------------
# 实体占位符映射表
# token → (分类, query 取值字段, {params 参数键: 记录字段})
# ---------------------------------------------------------------------------

ENTITY_TOKEN_MAP = {
    '[电站]': ('电站', 'name',
              {'siteId': 'id', 'powerStationId': 'id', 'powerStationIds': 'id',
               'ids': 'id', 'siteIds': 'id', 'stationId': 'id'}),
    '[项目]': ('项目', 'name',
              {'settlementOrganizationId': 'id'}),
    '[逆变器设备SN]': ('Inverter', 'deviceSn',
                      {'deviceSn': 'deviceSn', 'externalId_like': 'deviceSn',
                       'deviceId': 'id'}),
    '[逆变器设备ID]': ('Inverter', 'id',
                      {'deviceId': 'id'}),
    '[逆变器设备名称]': ('Inverter', 'name',
                       {'deviceName_like': 'name'}),
    '[气象站设备SN]': ('MeteorologicalStation', 'deviceSn',
                      {'deviceSn': 'deviceSn', 'deviceId': 'id'}),
    '[气象站设备ID]': ('MeteorologicalStation', 'id',
                      {'deviceId': 'id'}),
    # 执行人跨消缺工单/消缺记录/巡检工单三个分类（字段名不同，query 取值 fallback）
    '[执行人]': (('消缺工单', '消缺记录', '巡检工单'), ('chargePerson', 'chargePersonName'),
                {'chargePersonId': 'chargePersonId'}),
    '[整改人]': ('隐患工单', 'rectifyPersonName',
                {'rectifyPersonId': 'rectifyPersonId'}),
    '[修改人]': ('低效告警白名单', 'changedBy',
                {'changedUserId': 'changedBy', 'changedBy': 'changedBy'}),
    '[消缺工单编号]': ('消缺工单', 'id', {'number': 'id'}),
    '[巡检工单编号]': ('巡检工单', 'id', {'number': 'id'}),
    '[消缺记录工单编号]': ('消缺记录', 'id', {'number': 'id'}),
}

# 区域类占位符：从电站记录的 provinceCityDistrict 派生（'-' 分隔取前 N 段，99 = 全量）
_REGION_TOKENS = {
    '[大区/省]': 1,
    '[大区/省-州/市]': 2,
    '[大区/省-州/市-区]': 3,
    '[地址]': 99,
}

# 关联规则：锚实体（电站）与其他分类的外键关系
# 分类 → (记录字段, 匹配方式)；None 表示无法关联，全局随机
_STATION_LINK_FIELDS = {
    'Inverter': ('siteId', 'eq'),
    'MeteorologicalStation': ('siteId', 'eq'),
    '消缺工单': ('powerStationIds', 'in'),
    '消缺记录': ('powerStationIds', 'in'),
    '巡检工单': ('powerStationId', 'eq'),
    '隐患工单': ('powerStationId', 'eq'),
    '低效告警白名单': ('stationId', 'eq'),
    '项目': None,
    '电站': None,
}


def load_entity_mapping(mapping_path: str | Path) -> dict[str, list[dict]]:
    """加载实体映射文件，返回 {分类: [记录, ...]}"""
    path = Path(mapping_path)
    if not path.exists():
        raise FileNotFoundError(
            f'[占位符填充] 实体映射文件不存在: {path}，'
            f'请检查 dataset.placeholder_fill.entity_mapping_path 配置'
        )
    mapping = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f'[占位符填充] 实体映射文件格式错误或为空: {path}')
    return mapping


class _FillerContext:
    """单行填充上下文：持有随机源、基准时间、锚实体与该行已抽取实体缓存"""

    def __init__(self, mapping: dict, rng: random.Random, now: datetime,
                 source_info: str):
        self.mapping = mapping
        self.rng = rng
        self.now = now
        self.source_info = source_info
        self.anchor = None          # 锚实体（电站记录）
        self._cache: dict[str, dict] = {}   # token → 已抽取的记录
        self._by_category: dict[str, dict] = {}  # 分类 → 记录（同分类复用同一记录）

    # -- 记录抽取 ----------------------------------------------------------

    def _linked_records(self, category: str) -> list[dict] | None:
        """返回锚实体关联的记录池；无锚或无法关联时返回 None"""
        if self.anchor is None:
            return None
        rule = _STATION_LINK_FIELDS.get(category)
        if rule is None:
            return None
        field, mode = rule
        pool = self.mapping.get(category, [])
        if mode == 'eq':
            return [r for r in pool if str(r.get(field, '')) == str(self.anchor.get('id', ''))]
        if mode == 'in':
            return [r for r in pool
                    if str(self.anchor.get('id', '')) in str(r.get(field, '')).split(',')]
        return None

    def _pick_record(self, categories) -> dict:
        """按分类抽取记录：优先锚关联池（各分类独立筛选后合并），
        关联池为空或无法关联时全局随机"""
        if isinstance(categories, str):
            categories = (categories,)
        pool: list[dict] = []
        for cat in categories:
            cat_pool = self.mapping.get(cat, [])
            linked = self._linked_records(cat)
            if linked:
                pool.extend(linked)
            elif self.anchor is not None and _STATION_LINK_FIELDS.get(cat):
                logger.warning(
                    f'[占位符填充] 锚电站(id={self.anchor.get("id")}) 下无关联'
                    f'"{cat}" 记录，降级全局随机，位置: {self.source_info}'
                )
                pool.extend(cat_pool)
            else:
                pool.extend(cat_pool)
        if not pool:
            raise ValueError(
                f'[占位符填充] 实体映射中分类 {categories} 无任何记录，'
                f'位置: {self.source_info}'
            )
        return self.rng.choice(pool)

    def record_for(self, token: str, categories) -> dict:
        """同一行内：同一 token 复用同一记录；同一分类（组）复用同一记录。

        例如 [逆变器设备SN] / [逆变器设备ID] / [逆变器设备名称] 必须指向
        同一台逆变器，[执行人] 与 [消缺工单编号] 必须来自同一条工单，
        否则 query 与 expected_parameters 描述的实体不一致。
        """
        record = self._by_category.get(categories)
        if record is None:
            record = self._pick_record(categories)
            self._by_category[categories] = record
        self._cache[token] = record
        if token == '[电站]' and self.anchor is None:
            self.anchor = record
        return record

    # -- 取值 --------------------------------------------------------------

    def value_for(self, token: str, param_key: str | None) -> str:
        """按上下文取占位符的值：params 按参数键取字段，query 取可读字段"""
        if token in _REGION_TOKENS:
            return self._region_value(token)
        categories, query_fields, param_fields = ENTITY_TOKEN_MAP[token]
        record = self.record_for(token, categories)
        fields = query_fields if isinstance(query_fields, (tuple, list)) else (query_fields,)
        if param_key is not None:
            mapped = param_fields.get(param_key)
            if mapped is not None:
                fields = (mapped,)
        value = next((record.get(f) for f in fields if record.get(f) is not None), None)
        if value is None:
            logger.warning(
                f'[占位符填充] 记录缺少字段 {fields}（token={token}, '
                f'key={param_key}），用空值替换，位置: {self.source_info}'
            )
            value = ''
        return str(value)

    def _region_value(self, token: str) -> str:
        """从电站记录派生区域值：provinceCityDistrict 按 '-' 取前 N 段"""
        record = self.record_for(token, '电站')
        raw = str(record.get('provinceCityDistrict', '') or '')
        parts = raw.split('-')
        n = _REGION_TOKENS[token]
        if n >= len(parts):
            return raw
        return '-'.join(parts[:n])


def _extract_param_key(text: str, pos: int) -> str | None:
    """取 params 文本中占位符所属的参数键：往前找最近的 ';' 与 '='"""
    seg_start = text.rfind(';', 0, pos)
    seg = text[seg_start + 1:pos]
    eq = seg.rfind('=')
    if eq == -1:
        return None
    return seg[:eq].strip()


def _fill_text(text: str, ctx: _FillerContext, is_params: bool,
               column: str) -> str:
    """填充单个文本中的全部占位符；未识别的占位符 fail fast"""
    if not text or '[' not in text:
        return text

    def repl(match: re.Match) -> str:
        token = match.group(0)
        # 剥离反引号包裹，仅保留 [xxx] 内部 token 用于分类判定
        if len(token) >= 3 and token[0] == '`' and token[-1] == '`':
            token = token[1:-1]
        if token in _TIME_TOKENS:
            return _time_value(token, ctx.now)
        rel = _REL_TIME_RE.match(token)
        if rel:
            return _rel_time_value(rel, ctx.now)
        if token in ENTITY_TOKEN_MAP or token in _REGION_TOKENS:
            param_key = _extract_param_key(text, match.start()) if is_params else None
            return ctx.value_for(token, param_key)
        raise ValueError(
            f'[占位符填充] 无法识别的占位符: {token}\n'
            f'位置: {ctx.source_info}（{column}列）\n'
            f'请将 {token} 加入实体映射（entity_mapping.json）或时间占位符规则'
            f'（固定写法或 [N天前起始]/[N个月前起始]/[N年前起始] 泛化写法，'
            f'可加"结束"）后重试'
        )

    return TOKEN_RE.sub(repl, text)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def _match_target(source_path: str, target_dirs: list[str]) -> bool:
    """判断用例源路径是否命中填充目标目录（路径片段匹配）"""
    norm = source_path.replace('\\', '/')
    for d in target_dirs:
        d_norm = d.strip().strip('/').replace('\\', '/')
        if not d_norm:
            continue
        if f'/{d_norm}/' in norm or norm.endswith(f'/{d_norm}'):
            return True
    return False


def _is_under(path: str, dir_path: str) -> bool:
    """判断 path 是否位于 dir_path 目录内（绝对路径前缀匹配）"""
    path_abs = os.path.abspath(path)
    dir_abs = os.path.abspath(dir_path)
    return path_abs == dir_abs or path_abs.startswith(dir_abs + os.sep)


def fill_test_cases(test_cases_by_class: dict, seed: int | None = None,
                    conf=None) -> int | None:
    """对命中 target_dirs 的测试数据集执行占位符填充（原地修改行 dict）。

    :param test_cases_by_class: make_test_case_list 返回的 {类名: [test_case, ...]}
    :param seed: 随机种子；None 时自动生成并返回实际使用的 seed
    :param conf: ConfigReader 实例，None 时获取单例
    :return: 实际使用的 seed（未启用填充或未命中时返回入参 seed）
    """
    conf = conf or ConfigReader.get_instance()
    pf_conf = conf.get('dataset.placeholder_fill', {})
    if not pf_conf.get('enabled', False):
        logger.debug('[占位符填充] 未启用（dataset.placeholder_fill.enabled=false），跳过')
        return seed

    mapping_path = pf_conf.get('entity_mapping_path')
    if not mapping_path:
        raise ValueError(
            '[占位符填充] 未配置 dataset.placeholder_fill.entity_mapping_path'
        )
    target_dirs = pf_conf.get('target_dirs', [])
    if not target_dirs:
        raise ValueError(
            '[占位符填充] 未配置 dataset.placeholder_fill.target_dirs'
        )
    output_dir = pf_conf.get('output_dir')

    mapping = load_entity_mapping(mapping_path)
    if seed is None:
        seed = random.randrange(1, 2**31)
    rng = random.Random(seed)
    now = datetime.now()

    logger.info(f'[占位符填充] 启用，seed={seed}，实体映射: {mapping_path}，'
                f'目标目录: {target_dirs}，输出目录: {output_dir}')

    filled_count = 0
    replaced_count = 0
    hit_sources = []
    for class_name, class_cases in test_cases_by_class.items():
        for tc in class_cases:
            source = tc['case_name']
            if not _match_target(source, target_dirs):
                continue
            if output_dir and _is_under(source, output_dir) \
                    and not _match_target(source, target_dirs):
                logger.info(
                    f'[占位符填充] {source} 位于填充输出目录内'
                    f'（已填充过的数据集），跳过填充'
                )
                continue
            hit_sources.append(source)
            rows = tc['csv']
            ctx = _FillerContext(mapping, rng, now, source)
            for idx, row in enumerate(rows):
                # 每行独立随机：行级上下文（含锚实体与 token 缓存）
                row_ctx = _FillerContext(mapping, rng, now,
                                         f'{source} 第{idx + 2}行 '
                                         f'(用例:{row.get("id", row.get("用例编号", "?"))})')
                for column in ('query', 'expected_parameters', 'actual_capability_id'):
                    raw = row.get(column, '')
                    if '[' in raw:
                        filled = _fill_text(raw, row_ctx, column == 'expected_parameters',
                                            column)
                        replaced_count += len(TOKEN_RE.findall(raw))
                        row[column] = filled
            filled_count += len(rows)
            logger.info(f'[占位符填充] 已完成: {source} ({len(rows)} 行)')
            # 产物按原文件名写入输出目录（已存在同名文件时覆盖）
            if output_dir:
                out_path = _write_output_csv(source, rows, output_dir)
                logger.info(f'[占位符填充] 产物已输出: {out_path}')

    if hit_sources:
        logger.info(f'[占位符填充] 汇总: {len(hit_sources)} 个数据集, '
                    f'{filled_count} 行, 替换 {replaced_count} 处占位符, seed={seed}')
    else:
        logger.info(f'[占位符填充] 本次运行未命中任何 target_dirs 数据集，未做替换')
    return seed


def fill_raw_datasets(seed: int | None = None, conf=None) -> int | None:
    """嗅探 raw 目录（target_dirs 命中的路径片段）下的待填充模板，填充后写入 output_dir。

    与评测嗅探分离：本函数只负责"模板 → 产物"的生成阶段，
    供 main.py 嗅探路径在评测嗅探之前调用（先替换，再评测可执行数据集）。

    :param seed: 随机种子，None 时自动生成并返回实际使用的 seed
    :param conf: ConfigReader 实例，None 时获取单例
    :return: 实际使用的 seed（未启用填充或未找到模板时返回入参 seed）
    """
    conf = conf or ConfigReader.get_instance()
    pf_conf = conf.get('dataset.placeholder_fill', {})
    if not pf_conf.get('enabled', False):
        logger.debug('[占位符填充] 未启用（dataset.placeholder_fill.enabled=false），跳过')
        return seed

    mapping_path = pf_conf.get('entity_mapping_path')
    target_dirs = pf_conf.get('target_dirs', [])
    output_dir = pf_conf.get('output_dir')
    default_path = conf.get('dataset.default_dataset_path', None)
    if not output_dir:
        raise ValueError('[占位符填充] 未配置 dataset.placeholder_fill.output_dir')
    if not default_path or not Path(default_path).is_dir():
        raise ValueError(f'[占位符填充] 默认数据集目录无效: {default_path}')

    mapping = load_entity_mapping(mapping_path)
    if seed is None:
        seed = random.randrange(1, 2**31)
    rng = random.Random(seed)
    now = datetime.now()

    # 递归扫描 default_dataset_path 下命中 target_dirs 片段的待填充模板
    template_files = [
        str(p) for p in Path(default_path).rglob('test_cases_*.csv')
        if _match_target(str(p), target_dirs)
    ]
    if not template_files:
        logger.info(
            f'[占位符填充] raw 目录（target_dirs={target_dirs}）下未找到待填充模板，'
            f'跳过生成阶段'
        )
        return seed

    from tool.csv_reader import CsvReader

    total_rows = 0
    total_replaced = 0
    for src in sorted(template_files):
        reader = CsvReader(src)
        rows = reader.read_rows()
        for idx, row in enumerate(rows):
            row_ctx = _FillerContext(
                mapping, rng, now,
                f'{src} 第{idx + 2}行 (用例:{row.get("id", row.get("用例编号", "?"))})'
            )
            for column in ('query', 'expected_parameters', 'actual_capability_id'):
                raw_text = row.get(column, '')
                if '[' in raw_text:
                    total_replaced += len(TOKEN_RE.findall(raw_text))
                    row[column] = _fill_text(raw_text, row_ctx,
                                             column == 'expected_parameters', column)
        out_path = _write_output_csv(src, rows, output_dir)
        total_rows += len(rows)
        logger.info(f'[占位符填充] 模板已填充并输出: {src} → {out_path} ({len(rows)} 行)')

    logger.info(f'[占位符填充] raw 生成阶段汇总: {len(template_files)} 个模板, '
                f'{total_rows} 行, 替换 {total_replaced} 处, seed={seed}')
    return seed


def _write_output_csv(source_path: str, rows: list[dict], output_dir: str) -> Path:
    """将填充后的行按原文件名写入输出目录，返回输出路径（已存在则覆盖）"""
    from tool.csv_writer import CsvWriter

    out_dir = Path(output_dir)
    out_path = out_dir / Path(source_path).name
    fieldnames = list(rows[0].keys()) if rows else None
    CsvWriter(out_path).write_rows(rows, fieldnames=fieldnames)
    return out_path


# ---------------------------------------------------------------------------
# 填充预览（--fill-preview）
# ---------------------------------------------------------------------------

def preview_fill(csv_path: str | None, seed: int | None = None,
                 conf=None) -> int | None:
    """仅填充并输出 *.filled.csv，不执行评测。

    :param csv_path: 指定 CSV 文件；None 时嗅探扫描 target_dirs 命中的所有数据集
    :param seed: 随机种子，None 时自动生成
    :return: 实际使用的 seed
    """
    conf = conf or ConfigReader.get_instance()
    pf_conf = conf.get('dataset.placeholder_fill', {})
    target_dirs = pf_conf.get('target_dirs', [])
    output_dir = pf_conf.get('output_dir')
    if not output_dir:
        raise ValueError(
            '[填充预览] 未配置 dataset.placeholder_fill.output_dir'
        )

    # ---- 加载目标行 ----
    targets: list[tuple[str, list[dict], list[str]]] = []  # (path, rows, headers)
    if csv_path is not None:
        from tool.csv_reader import CsvReader
        if not (Path(csv_path).exists() and Path(csv_path).suffix.lower() == '.csv'):
            raise ValueError(f'[填充预览] 无效的 CSV 文件路径: {csv_path}')
        if _is_under(csv_path, output_dir) and not _match_target(csv_path, target_dirs):
            raise ValueError(
                f'[填充预览] {csv_path} 位于填充输出目录内且不在 raw 源目录中，'
                f'请指定未填充的源数据集（{output_dir} 下的文件已填充过）'
            )
        reader = CsvReader(csv_path)
        targets.append((csv_path, reader.read_rows(), reader.get_headers()))
        if not _match_target(csv_path, target_dirs):
            logger.warning(
                f'[填充预览] {csv_path} 不在 placeholder_fill.target_dirs 命中范围内，'
                f'正常评测时不会被填充'
            )
    else:
        default_path = conf.get('dataset.default_dataset_path', None)
        if not default_path or not Path(default_path).is_dir():
            raise ValueError(
                f'[填充预览] 未指定 --csv_path 且默认数据集目录无效: {default_path}'
            )
        # 递归扫描命中 target_dirs（raw 源目录）的模板；
        # 输出目录内的已填充产物不含 raw 路径片段，自然被过滤
        for file in sorted(Path(default_path).rglob('test_cases_*.csv')):
            src = str(file)
            if not _match_target(src, target_dirs):
                continue
            from tool.csv_reader import CsvReader
            reader = CsvReader(src)
            targets.append((src, reader.read_rows(), reader.get_headers()))

    if not targets:
        raise ValueError(
            f'[填充预览] 未找到任何目标数据集（target_dirs={target_dirs}）'
        )

    mapping_path = pf_conf.get('entity_mapping_path')
    mapping = load_entity_mapping(mapping_path)
    if seed is None:
        seed = random.randrange(1, 2**31)
    rng = random.Random(seed)
    now = datetime.now()

    for src, rows, headers in targets:
        total_replaced = 0
        for idx, row in enumerate(rows):
            row_ctx = _FillerContext(mapping, rng, now,
                                     f'{src} 第{idx + 2}行 '
                                     f'(用例:{row.get("id", row.get("用例编号", "?"))})')
            for column in ('query', 'expected_parameters', 'actual_capability_id'):
                raw = row.get(column, '')
                if '[' in raw:
                    total_replaced += len(TOKEN_RE.findall(raw))
                    row[column] = _fill_text(raw, row_ctx,
                                             column == 'expected_parameters', column)

        # 产物按原文件名写入输出目录（已存在同名文件时覆盖）
        out_path = _write_output_csv(src, rows, output_dir)
        logger.info(f'[填充预览] 已输出: {out_path} ({len(rows)} 行, '
                    f'替换 {total_replaced} 处)')
        print(f'[填充预览] 已输出: {out_path} ({len(rows)} 行, '
              f'替换 {total_replaced} 处, seed={seed})')

    # ---- 控制台打印替换对比样例 ----
    sample_shown = 0
    for src, rows, _ in targets:
        for row in rows:
            query = row.get('query', '')
            if '[' not in query:
                continue
            # 从原始文本无法直接取"替换前"，此处打印填充后样例即可
            print(f'[填充预览] 样例 ({Path(src).name} '
                  f'{row.get("id", row.get("用例编号", "?"))}): {query}')
            sample_shown += 1
            if sample_shown >= 5:
                return seed
    return seed


if __name__ == '__main__':
    """独立调试入口: python -m pipeline.placeholder_filler --csv <path> [--seed N]"""
    import argparse

    parser = argparse.ArgumentParser(description='占位符填充预览（独立调试）')
    parser.add_argument('--csv', type=str, default=None, help='测试数据集 CSV 路径')
    parser.add_argument('--seed', type=int, default=None, help='随机种子')
    args = parser.parse_args()

    preview_fill(args.csv, args.seed)
