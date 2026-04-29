import csv
from pathlib import Path
from typing import Any, Dict, List, Optional


class CsvReader:
    """用于读取和解析 CSV 文件的类"""

    def __init__(self, csv_path: str | Path):
        """
        初始化 CSV 读取器

        :params: csv_path: CSV 文件路径
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {self.csv_path}")

    def read_rows(self) -> List[Dict[str, str]]:
        """将 CSV 解析为字典列表，每一行对应一个字典"""
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [dict(row) for row in reader]

    def read_by_key(self, key_field: str) -> Dict[str, Dict[str, str]]:
        """将 CSV 解析为字典，使用指定列作为顶层 key"""
        rows = self.read_rows()
        result: Dict[str, Dict[str, str]] = {}

        for row in rows:
            if key_field not in row:
                raise KeyError(f"CSV 中不存在字段: {key_field}")
            key = row[key_field]
            result[key] = row

        return result

    def get_headers(self) -> List[str]:
        """获取 CSV 表头"""
        with open(self.csv_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return reader.fieldnames or []


if __name__ == "__main__":
    sample_path = Path(__file__).parent.parent / "test_suite" / "test_cases.csv"
    reader = CsvReader(sample_path)

    rows = reader.read_rows()
    print("前两行字典:")
    print(rows[:2])

    keyed_rows = reader.read_by_key("id")
    print("按 id 建立字典后的示例:")
    first_key = next(iter(keyed_rows))
    print(first_key, keyed_rows[first_key])
