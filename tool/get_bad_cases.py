"""
失败用例提取工具

扫描评测结果目录下所有 result_outputs_*.csv 文件，
提取 is_success 为空字符串或 'False' 的行，
输出到同目录下的 bad_cases_<suffix>.csv。
"""

import os

from tool.csv_reader import CsvReader
from tool.csv_writer import CsvWriter


def _is_failed(value: str) -> bool:
    """
    CSV 读回的值均为字符串。
    is_success 为空字符串或 'False'（忽略大小写）视为失败。
    """
    return value.strip().lower() in ('', 'false')


def extract_bad_cases(base_path: str) -> dict[str, tuple[int, int]]:
    """
    扫描 base_path 下所有 result_outputs_*.csv，
    将失败行提取到 bad_cases_<suffix>.csv

    :param base_path: 结果输出目录路径
    :return: {suffix: (bad_count, total_count), ...}
    """
    if not os.path.isdir(base_path):
        print(f"[bad_cases] 目录不存在: {base_path}")
        return {}

    stats: dict[str, tuple[int, int]] = {}

    for file in sorted(os.listdir(base_path)):
        if not (file.startswith('result_outputs_') and file.endswith('.csv')):
            continue

        # result_outputs_normal.csv → normal
        suffix = file[len('result_outputs_'):].removesuffix('.csv')

        csv_path = os.path.join(base_path, file)
        rows = CsvReader(csv_path).read_rows()

        bad_rows = [row for row in rows if _is_failed(row.get('is_success', ''))]
        bad_count = len(bad_rows)
        total_count = len(rows)

        stats[suffix] = (bad_count, total_count)

        if bad_rows:
            bad_file = f'bad_cases_{suffix}.csv'
            bad_path = os.path.join(base_path, bad_file)
            CsvWriter(bad_path).write_rows(bad_rows)
            print(f"[bad_cases] {bad_file}: {bad_count}/{total_count} 条失败用例已提取")
        else:
            print(f"[bad_cases] {suffix}: 0/{total_count} 条失败用例，跳过")

    total_bad = sum(s[0] for s in stats.values())
    total_all = sum(s[1] for s in stats.values())
    print(f"[bad_cases] 总计: {total_bad}/{total_all} 条失败用例")

    return stats
