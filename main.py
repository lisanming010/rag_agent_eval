from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from deepeval.test_case import LLMTestCase
from functools import partial
from datetime import datetime
import argparse
import os

from agents.http_agent import HTTPAgent
from tool import AsyncResultWriter, CsvReader, CsvWriter, ConfigReader
from evaluator.metrics import reverse_validation_metric, contextual_recall_metric
from tool.collection_result import CollectionResult


TEST_SUITE_MAP = {}

METRICS_MAP = {
    'reverse_validation': reverse_validation_metric,
    'contextual_recall': contextual_recall_metric
}

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="执行评测套件")
    parser.add_argument(
        '-cp',
        '--csv_path',
        type=str,
        help='可以指定测试数据集csv文件路径，需要指定文件名，不指定默认执行test_suite下所有csv格式的测试用例集。'
    )
    parser.add_argument(
        '-m',
        '--metrics',
        choices=['reverse_validation', 'contextual_recall'],
        nargs='+',
        help='指定评测指标，默认为reverse_validation，可以选择reverse_validation和contextual_recall两种指标，单独使用或同时使用两者进行评测。指定测试数据集路径时该参数必传' 
    )
    return parser.parse_args()

def csv_2_case_dict(csv_path, metrics:list)->dict:
    """
    从csv文件创建测试用例
    
    :params: csv_path: csv文件路径
    :params: metrics: 指定使用的metrics
    """
    test_cases = {}
    test_cases['csv'] = CsvReader(csv_path).read_rows()
    test_cases['case_name'] = str(csv_path)
    test_cases['metrics'] = metrics
    return test_cases

def make_test_case_list(csv_path, metrics:list|None)->list[dict]:
    """
    构建测试数据集

    :params: csv_path:csv文件的路径
    :metrics: 传递的metrics
    """

    print('读取测试用例集中......')
    test_case_list = []
    # 测试数据集转换
    if csv_path is not None:
        if os.path.exists(csv_path) and os.path.splitext(csv_path)[1].lower() == '.csv':
            if not metrics:
                raise ValueError(f'指定测试数据集时-m参数必传')
            test_case_list.append(csv_2_case_dict(csv_path, metrics))
        else:
            raise ValueError(f"提供的csv文件路径无效或非csv格式文件: {csv_path}")
    else:
        default_csv_path = conf_reader.get('dataset.default_dataset_path', None)
        if default_csv_path is None:
            raise ValueError(f'执行命令中未指定测试数据集文件，配置文件中也无对应配置，请检查配置文件dataset.default_dataset_path配置项')
    
        if os.path.exists(default_csv_path):
            # 配置文件中指定的是文件夹，默认在当前层级下寻找csv格式文件
            if os.path.isdir(default_csv_path):
                for file in os.listdir(default_csv_path):
                    # 检索test_cases_xxx.csv文件
                    if file.endswith('.csv'):
                        data_set_type = file.removesuffix('.csv').split('_')[-1]
                        if 'test_cases_' not in file:
                            print(f'{file}未被匹配')
                            continue
                        metrics = conf_reader.get(f'dataset.dataset_metrics_map.{data_set_type}', None)
                        if metrics is None:
                            raise ValueError(f'{default_csv_path}{file}文件自动匹配metrics方法失败，请检查文件名称是否正确')
                        csv_path = os.path.join(default_csv_path, file)
                        test_case_list.append(csv_2_case_dict(csv_path, metrics))
                if test_case_list == []:
                    raise ValueError(f'未匹配到测试数据集')
            # 配置文件中指定特定的csv文件
            elif os.path.splitext(default_csv_path)[1].lower() == '.csv':
                metrics = conf_reader.get('dataset.metrics_if_specify_csv', None)
                if metrics is None:
                    raise ValueError(f'配置文件中未指定测试数据集使用的metrics')
                test_case_list.append(csv_2_case_dict(default_csv_path, metrics))
        else:
            raise ValueError(f'测试数据集：{default_csv_path}目录不存在')
    
    return test_case_list

def call_agent(agent:HTTPAgent, test_case_csv:dict):
    """agent调用入口，将agent响应并入原有字典中"""

    _, answer_summary, res_time = agent.call_agent(test_case_csv['question'])
    test_case_csv['agent_response'] = answer_summary
    test_case_csv['res_time(s)'] = res_time

def run_in_thread_pool(fn, items:list, max_workers:int=8, task_name:str="task"):
    """
    通用线程池调度方法，对items列表中的每个元素并发执行fn

    :params: fn: 可调用对象，接受单个item作为参数，可通过functools.partial绑定额外参数
    :params: items: 待处理的数据列表
    :params: max_workers: 最大线程数，默认8
    :params: task_name: 任务名称，用于日志输出
    """
    total = len(items)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in as_completed(futures):
            item = futures[future]
            completed += 1
            try:
                future.result()
                print(f"[{completed}/{total}] {task_name}成功")
            except Exception as e:
                print(f"[{completed}/{total}] {task_name}失败: {e}")

# TODO:llm_test_case是否应该再单独拆出来一个字典
def make_llm_case(test_case_csv:dict)->LLMTestCase:
    """
    llm_case组装方法,组装完成的用例重新放回字典中

    :params: test_case_csv:测试数据集，即每个csv文件的一列
    """
    llm_test_case = LLMTestCase(
        input=test_case_csv['question'],
        expected_output=test_case_csv['expected_answer'],
        context=test_case_csv['negative_criteria'].split('|'),
        retrieval_context=[test_case_csv['agent_response']]
    )
    test_case_csv['llm_test_case'] = llm_test_case

def run_evaluate(test_case:dict):
    """
    执行evaluate，并将执行结果写回用例字典中

    :params: test_case:测试数据集,即测试用例list中的整个成员，test_case_list[0]
    """

    conf_reader = ConfigReader.get_instance()

    
    test_case_csv = test_case['csv']
    # 提取字典中所有llmtestcase
    llm_test_case = [case['llm_test_case'] for case in test_case_csv]
    metrics_str = test_case['metrics']
    metrics = []
    for metric in metrics_str:
        eva_metric = METRICS_MAP.get(metric, None)
        if eva_metric is None:
            raise ValueError(f'{metric}尚未注册，请在evaluator/metrics.py中实现后在main中完成注册')
        metrics.append(eva_metric)

    eva_run_async = conf_reader.get('evluate.run_async')
    eva_max_concurrent = conf_reader.get('evluate.max_concurrent')
    eva_throttle_value = conf_reader.get('evluate.throttle_value')
    result = evaluate(
        llm_test_case,
        metrics,
        async_config=AsyncConfig(
            run_async=eva_run_async,
            max_concurrent=eva_max_concurrent,
            throttle_value=eva_throttle_value
        )
    )

    # zip做测试用例和测试结果关联，evaluate能够确保执行结果和llm_test_case顺序对应
    for case_dict, test_result in zip(test_case_csv, result.test_results):
        case_dict['is_success'] = test_result.success
        for md in test_result.metrics_data:
            metrics_name = md.name
            case_dict[f'{metrics_name}_is_success'] = md.success
            case_dict[f'{metrics_name}_score'] = md.score
            case_dict[f'{metrics_name}_threshold'] = md.threshold
            case_dict[f'{metrics_name}_reason'] = md.reason


def mkdir_with_timestamp(base_path):
    """
    创建当前时间戳命名的文件夹,父目录不存在的情况下也会直接创建出父目录,返回父目录+时间戳目录

    :params: base_path:父目录
    """
    timestamp = datetime.now().strftime('%m%d%H%M%S')
    output_dir = os.path.join(base_path, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

if __name__ == "__main__":
    conf_reader = ConfigReader.get_instance()
    params = parse_args()
    csv_path = params.csv_path
    metrics = params.metrics

    test_cases_list = make_test_case_list(csv_path, metrics)

    # 调用agent，用例中添加agent返回响应
    agent_type = conf_reader.get('agents.type')
    if agent_type == 'http':
        agent_endpoint = conf_reader.get('agents.http_agent.endpoint')
        agent_search_mode = conf_reader.get('agents.http_agent.mode')
        agent = HTTPAgent(agent_endpoint, agent_search_mode)
        all_cases = [
            test_case
            for cases in test_cases_list
            for test_case in cases['csv']
        ]
        
        max_worker = conf_reader.get("agents.http_agent.call_agent_th_max")

        # 调用agent获取响应
        run_in_thread_pool(partial(call_agent, agent), all_cases, max_workers=max_worker, task_name="call_agent")
        # 组装llm_test_case
        run_in_thread_pool(make_llm_case, all_cases, max_workers=max_worker, task_name='make_llm_test_case')
        
        # 执行断言并回写测试结果
        result_csv_path_list = []
        result_save_path = conf_reader.get("result.save_path")
        base_path = mkdir_with_timestamp(result_save_path)
        # 异步写入
        writer = AsyncResultWriter(base_path)
        writer.start()

        for test_case in test_cases_list:
            run_evaluate(test_case)
            writer.submit(test_case, test_case['case_name'])
        writer.wait_and_stop()
        print(writer.get_stats())

        # 汇总测试结果
        for file in os.listdir(base_path):
            if file.endswith('.csv') and 'result_outputs_' in file:
                result_csv_path = os.path.join(base_path, file)
                collection_result = CollectionResult(result_csv_path)

                # 任务执行结果汇总统计
                task_success_stat = collection_result.task_success_stats()
                print("\n========任务执行结果统计==========")
                for k, v in task_success_stat:
                    print(f'{k}: {v}')
                
                # 响应时间结果汇总统计
                res_time = collection_result.res_time()
                print("\n========响应时间统计==========")
                for k, v in res_time:
                    print(f'{k}: {v}')

                        
    else:
        # TODO: 非HTTP调用的agent接入注册位置
        pass
