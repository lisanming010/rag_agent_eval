"""CSV 批次提交协议：数据先同步，检查点后提交；历史文件只读。"""

import csv
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tool.csv_writer import CsvWriter


class CheckpointError(ValueError):
    """历史结果不能作为可信恢复依据。"""


def checkpoint_path(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix('.checkpoint.json')


def csv_value(value) -> str:
    return '' if value is None else str(value)


def _digest_row(hasher, row: dict, columns: list[str]):
    # 与 CSV 的字符串序列化一致；列顺序以 checkpoint 为准，允许后续扩展表头。
    payload = [csv_value(row.get(column)) for column in columns]
    hasher.update(json.dumps(payload, ensure_ascii=False, separators=(',', ':')).encode('utf-8'))
    hasher.update(b'\n')


def _decoded_lines(source):
    # 不使用 TextIOWrapper 的预读解码：未提交尾部可能正好截断在 UTF-8 字符中间。
    for index, line in enumerate(source):
        yield line.decode('utf-8-sig' if index == 0 else 'utf-8')


def read_csv_prefix(path: Path, count: int | None = None) -> tuple[list[dict], bool]:
    """严格读取 N 条完整逻辑记录，不解析 N 条之后可能损坏的尾部。"""
    csv.field_size_limit(max(csv.field_size_limit(), 2 ** 31 - 1))
    with path.open('rb') as source:
        reader = csv.reader(_decoded_lines(source), strict=True)
        fields = next(reader, None)
        if not fields or any(not field for field in fields) or len(set(fields)) != len(fields):
            raise CheckpointError(f'CSV 表头无效或重复: {path}')
        rows = []
        while count is None or len(rows) < count:
            values = next(reader, None)
            if values is None:
                if count is not None:
                    raise CheckpointError(f'CSV 少于已提交的 {count} 条记录: {path}')
                break
            if len(values) != len(fields):
                raise CheckpointError(f'CSV 第 {len(rows) + 1} 条记录列数错误: {path}')
            rows.append(dict(zip(fields, values)))
        return rows, bool(source.read(1))


def atomic_json(path: Path, payload: dict):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='w', encoding='utf-8', dir=path.parent,
            prefix=f'.{path.name}.', suffix='.tmp', delete=False,
        ) as target:
            temporary = Path(target.name)
            json.dump(payload, target, ensure_ascii=False, indent=2)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class CommittedResult:
    rows: list[dict]
    has_uncommitted_tail: bool = False


def read_committed(csv_path: str | Path, agent: str) -> CommittedResult:
    path = Path(csv_path)
    marker = checkpoint_path(path)
    if path.exists() and not path.is_file():
        raise CheckpointError(f'结果 CSV 路径不是文件: {path}')
    if marker.exists() and not marker.is_file():
        raise CheckpointError(f'checkpoint 路径不是文件: {marker}')
    if not marker.is_file():
        if path.exists():
            raise CheckpointError(f'旧格式结果缺少 checkpoint，不能增量恢复: {path}')
        return CommittedResult([])
    try:
        state = json.loads(marker.read_text(encoding='utf-8'))
        if not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] != 1:
            raise ValueError('不支持的 checkpoint 格式版本')
        if state.get('result_file') != path.name or state.get('agent') != agent.casefold():
            raise ValueError('checkpoint 的文件/Agent 归属不符')
        count = state['committed_rows']
        batch = state['last_committed_batch']
        fields = state['committed_columns']
        digest = state['content_sha256']
        if type(count) is not int or count < 0 or type(batch) is not int or batch < 0:
            raise ValueError('提交计数无效')
        if not isinstance(fields, list) or any(not isinstance(k, str) or not k for k in fields):
            raise ValueError('提交字段无效')
        if len(set(fields)) != len(fields) or not isinstance(digest, str) or not re.fullmatch('[0-9a-f]{64}', digest):
            raise ValueError('提交字段或摘要无效')
        hasher = hashlib.sha256()
        if count == 0:
            if batch != 0 or fields or digest != hasher.hexdigest():
                raise ValueError('零批次检查点无效')
            # 初始化检查点先于首次 CSV 写入；该状态没有任何可复用记录。
            return CommittedResult([], path.exists() and path.stat().st_size > 0)
        if not fields or batch < 1 or batch > count or not path.is_file():
            raise ValueError('已提交的 CSV 缺失或批次信息无效')
        rows, tail = read_csv_prefix(path, count)
        if any(field not in rows[0] for field in fields):
            raise ValueError('已提交字段在 CSV 中缺失')
        for row in rows:
            _digest_row(hasher, row, fields)
        if hasher.hexdigest() != digest:
            raise ValueError('已提交内容摘要不一致')
        # 扩展表头后、检查点提交前可能中断；不采纳旧检查点未覆盖的新列。
        return CommittedResult([{key: row[key] for key in fields} for row in rows], tail)
    except (ValueError, KeyError, TypeError, OSError, csv.Error) as exc:
        raise CheckpointError(f'checkpoint 校验失败: {marker}: {exc}') from exc


class CheckpointWriter:
    """只写新的结果文件。稳定表头增量摘要，扩展表头时重新校验整个新版本。"""

    def __init__(self, path: str | Path, agent: str):
        self.path = Path(path)
        self.marker = checkpoint_path(self.path)
        self.agent = agent.casefold()
        self.columns = []
        self.count = 0
        self.batch = 0
        self.hasher = hashlib.sha256()
        self.failed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.marker.exists():
            raise FileExistsError(f'拒绝覆盖已有结果或 checkpoint: {self.path}')
        atomic_json(self.marker, self._state(self.hasher, [], 0, 0, ''))

    def _state(self, hasher, fields, count, batch, last_id):
        return {
            'version': 1, 'result_file': self.path.name, 'agent': self.agent,
            'last_committed_batch': batch, 'committed_rows': count,
            'last_case_id': last_id, 'committed_columns': fields,
            'content_sha256': hasher.hexdigest(),
        }

    def commit(self, rows: list[dict]):
        if self.failed:
            raise RuntimeError('上次提交失败，禁止继续写入该结果')
        if not rows:
            raise ValueError('不能提交空批次')
        # 一次确定字符串值，确保落盘内容与摘要一致（包括 None、bool、嵌套对象）。
        rows = [{key: csv_value(value) for key, value in row.items()} for row in rows]
        fields = list(dict.fromkeys(self.columns + [key for row in rows for key in row]))
        try:
            writer = CsvWriter(self.path)
            if self.count:
                writer.append_rows(rows)
            else:
                writer.write_rows(rows)
            with self.path.open('rb+') as data_file:
                os.fsync(data_file.fileno())
            if self.count and fields != self.columns:
                hasher = hashlib.sha256()
                saved_rows, _ = read_csv_prefix(self.path, self.count + len(rows))
                for row in saved_rows:
                    _digest_row(hasher, row, fields)
            else:
                hasher = self.hasher.copy()
                for row in rows:
                    _digest_row(hasher, row, fields)
            last_id = next((rows[-1][key] for key in ('test_id', '用例编号', 'case_id', 'id', '_parent_case_id')
                            if rows[-1].get(key)), '')
            atomic_json(self.marker, self._state(hasher, fields, self.count + len(rows), self.batch + 1, last_id))
        except BaseException:
            self.failed = True
            raise
        self.columns = fields
        self.count += len(rows)
        self.batch += 1
        self.hasher = hasher
