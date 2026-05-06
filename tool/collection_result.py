import os

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
            if "_is_success" in key:
                key.removesuffix('_is_success')
                metrics_name_list.append(key)
        return metrics_name_list           

    def task_success_stats(self):
        """
        统计task通过率，包含总任务通过率以及各个metrics的通过率,返回总的任务成功率以及各metrics的成功率
        """

        task_success_count = {'total':0}
        metrics = self.get_metrics()
        for metirc in metrics:
            task_success_count[metirc] = 0

        results_list = self.csv_reader.read_rows()
        total_task_count = len(results_list) 

        for result in results_list:
            if result['is_success'] == "TRUE":
                task_success_count['total'] += 1
            for metric in metrics:
                if result[f'{metric}_is_success'] == "TRUE":
                    task_success_count[metirc] += 1

        total_task_success_rate = 


        
        
    