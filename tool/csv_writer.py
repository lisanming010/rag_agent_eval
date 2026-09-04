import csv
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional


class CsvWriter:
    """用于将字典列表写入 CSV 文件的类"""

    def __init__(self, csv_path: str | Path):
        """
        初始化 CSV 写入器

        :params: csv_path: CSV 文件路径
        """
        self.csv_path = Path(csv_path)

    def write_rows(
        self,
        data: List[Dict[str, Any]],
        fieldnames: Optional[List[str]] = None,
        mode: str = 'w'
    ) -> None:
        """
        将字典列表写入 CSV 文件

        :params: data: 字典列表，每个字典代表一行
        :params: fieldnames: 指定列顺序，不指定则自动从第一行提取
        :params: mode: 写入模式，'w'覆盖写入，'a'追加写入
        """
        if not data:
            raise ValueError("数据列表不能为空")

        # 汇总所有行的全部字段名（避免首行缺失后续行独有的列被截断）
        if fieldnames is None:
            fieldnames_set: dict[str, None] = {}
            fieldnames = []
            for row in data:
                for key in row:
                    if key not in fieldnames_set:
                        fieldnames_set[key] = None
                        fieldnames.append(key)

        # 确保父目录存在
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        write_header = mode == 'w' or not self.csv_path.exists()

        with open(self.csv_path, mode, encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if write_header:
                writer.writeheader()
            writer.writerows(data)

    def append_rows(self, data: List[Dict[str, Any]]) -> None:
        """按已有表头追加；新增字段时流式扩展旧表并原子替换，避免批次错列。"""
        if not data:
            raise ValueError("数据列表不能为空")
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            self.write_rows(data)
            return

        with self.csv_path.open('r', encoding='utf-8-sig', newline='') as source:
            old_fields = next(csv.reader(source), [])
        fields = list(dict.fromkeys(old_fields + [key for row in data for key in row]))
        if fields == old_fields:
            self.write_rows(data, fieldnames=fields, mode='a')
            return

        # Agent 响应/评测理由可能超过 csv 默认的单字段 128 KiB 上限。
        csv.field_size_limit(max(csv.field_size_limit(), 2 ** 31 - 1))
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode='w', encoding='utf-8-sig', newline='',
                dir=self.csv_path.parent, prefix=f'.{self.csv_path.name}.',
                suffix='.tmp', delete=False,
            ) as target:
                temporary_path = Path(target.name)
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                with self.csv_path.open('r', encoding='utf-8-sig', newline='') as source:
                    writer.writerows(csv.DictReader(source))
                writer.writerows(data)
            os.replace(temporary_path, self.csv_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def write_from_test_cases(
        self,
        data: List[Dict[str, Any]],
        exclude_keys: Optional[List[str]] = None
    ) -> None:
        """
        将测试用例字典列表写入 CSV，自动过滤不可序列化的字段

        :params: data: 字典列表
        :params: exclude_keys: 需要排除的字段名列表
        """
        if not data:
            raise ValueError("数据列表不能为空")

        exclude = set(exclude_keys or [])
        # 过滤掉不可序列化的字段（如对象类型）
        filtered_data = []
        for row in data:
            filtered_row = {
                k: v for k, v in row.items()
                if k not in exclude and isinstance(v, (str, int, float, bool, type(None)))
            }
            filtered_data.append(filtered_row)

        self.write_rows(filtered_data)


if __name__ == "__main__":
    # 示例用法
    sample_data = [
        {'name': 'Alice', 'age': 25, 'city': 'Beijing'},
        {'name': 'Bob', 'age': 30, 'city': 'Shanghai'},
        {'name': 'Charlie', 'age': 35, 'city': 'Guangzhou'}
    ]

    output_path = Path(__file__).parent.parent / "output" / "sample_output.csv"
    writer = CsvWriter(output_path)
    writer.write_rows(sample_data)
    print(f"CSV 文件已写入: {output_path}")
