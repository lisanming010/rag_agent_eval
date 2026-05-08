import os
from typing import Dict, List


class MarkdownWriter:
    """Markdown 报告写入器，实例化时传入完整路径包含输出文件名"""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self._is_write_title = False
        parent_dir = os.path.dirname(self.output_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    def _write_title(self, lines:List):
        if not self._is_write_title:
            lines.append(f"# 评测报告")
            lines.append("")
            lines.append(f"生成时间: {self._now()}")
            lines.append("")
            self._is_write_title = True

    def write_report(
        self,
        case_name: str,
        task_success_stats: Dict[str, float],
        res_time_stats: Dict[str, float],
    ) -> None:
        """
        写入md

        :params: case_name测试数据集名称
        :params: task_success_stats指标通过率
        :params: res_time_stats任务执行时间汇总
        """

        lines: List[str] = []
        self._write_title(lines)

        lines.append(f"## 测试数据集：{case_name}")
        lines.append("### 任务执行结果统计")
        lines.append("")
        lines.append("| 指标 | 通过率 (%) |")
        lines.append("|------|-----------|")
        for metric, rate in task_success_stats.items():
            lines.append(f"| {metric} | {rate:.2f}% |")
        lines.append("")

        lines.append("### 响应时间统计")
        lines.append("")
        lines.append("| 指标 | 耗时 (s) |")
        lines.append("|------|---------|")
        for key, value in res_time_stats.items():
            lines.append(f"| {key} | {value:.2f} |")
        lines.append("")

        content = "\n".join(lines) + "\n"
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(content)

    @staticmethod
    def _now() -> str:
        from datetime import datetime

        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
