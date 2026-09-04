import os
import re
import numpy as np

from tool.csv_reader import CsvReader

class CollectionResult:
    def __init__(self, result_csv_path):
        self.result_csv_path = result_csv_path
        self.csv_reader = CsvReader(result_csv_path)

    def get_metrics(self)->list:
        """获取csv列表中metrics的名称"""

        metrics_name_list = []
        csv_header = self.csv_reader.get_headers()
        for key in csv_header:
            match = re.fullmatch(r'(.+)_is_success(?:\(t\d+\))?', key)
            if match and match.group(1) not in metrics_name_list:
                metrics_name_list.append(match.group(1))
        return metrics_name_list           

    def task_success_stats(self)->dict:
        """
        统计task通过率，包含总任务通过率以及各个metrics的通过率,返回总的任务成功率以及各metrics的成功率

        :returns: 分组计算的任务成功率，{'total':, 'metricxxx':,....}
        """

        task_success_count = {'total':0}
        metrics = self.get_metrics()
        for metric in metrics:
            task_success_count[metric] = 0

        results_list = self.csv_reader.read_rows()
        total_task_count = len(results_list)

        for result in results_list:
            # is_success 是整体标识（单轮直接取，多轮合并行从 _parent_all_pass 写入）。
            if str(result.get('is_success', '')).strip().upper() == 'TRUE':
                task_success_count['total'] += 1
            for metric in metrics:
                metric_key = f'{metric}_is_success'
                metric_val = result.get(metric_key)
                if metric_val not in (None, ''):
                    if str(metric_val).strip().upper() == 'TRUE':
                        task_success_count[metric] += 1
                else:
                    # 多轮合并行：检查 *_is_success(t1), *_is_success(t2) ... 全部通过才算通过
                    # CSV 联合表头包含其他用例的额外轮次；仅统计该用例自身轮次。
                    count = int(result.get('_total_turns') or 1) if result.get('_parent_case_id') else 1
                    turn_vals = [result.get(f'{metric_key}(t{i})', '') for i in range(1, count + 1)]
                    if turn_vals and all(str(v).strip().upper() == 'TRUE' for v in turn_vals):
                        task_success_count[metric] += 1

        task_success_rate = {
            key: round(count / total_task_count, 4)*100 if total_task_count else 0
            for key, count in task_success_count.items()
        }

        return task_success_rate

    def res_time(self):
        """
        call agent响应时间结果汇总
        """
        res_time_list = []
        res_time_result_dict = {}

        result_list = self.csv_reader.read_rows()
        for result in result_list:
            val = result.get('res_time(s)', '').strip()
            if not val:
                continue
            res_time_list.append(float(val))

        if not res_time_list:
            res_time_result_dict['任务最长耗时'] = 0
            res_time_result_dict['任务耗时平均值'] = 0
            res_time_result_dict['任务耗时p99'] = 0
            res_time_result_dict['任务耗时p95'] = 0
        else:
            res_time_result_dict['任务最长耗时'] = max(res_time_list)
            res_time_result_dict['任务耗时平均值'] = np.mean(res_time_list)
            res_time_result_dict['任务耗时p99'] = np.percentile(res_time_list, 99)
            res_time_result_dict['任务耗时p95'] = np.percentile(res_time_list, 95)

        return res_time_result_dict
        
