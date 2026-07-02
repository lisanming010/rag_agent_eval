import argparse
import os
from functools import partial

from deepeval.test_case import LLMTestCase

from tool.config_reader import ConfigReader
from tool.concurrency import run_in_thread_pool
from tool.file_utils import mkdir_with_timestamp
from tool import AsyncResultWriter, MarkdownWriter
from tool.collection_result import CollectionResult
from tool.get_bad_cases import extract_bad_cases
from tool.log_factory import LogFactory
from agents.factory import create_agent
from pipeline.test_case_loader import make_test_case_list
from evaluator.runner import run_evaluate, run_multimodel_reevaluate

logger = LogFactory.get_logger(__name__)

#CLI 
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="执行评测套件")
    parser.add_argument(
        '-cp',
        '--csv_path',
        type=str,
        help='可以指定测试数据集csv文件路径，需要指定文件名，'
             '不指定默认执行test_suite下所有csv格式的测试用例集。'
    )
    parser.add_argument(
        '-m',
        '--metrics',
        choices=['reverse_validation', 'contextual_recall'],
        nargs='+',
        help='指定评测指标，默认为reverse_validation，'
             '可以选择reverse_validation和contextual_recall两种指标，'
             '单独使用或同时使用两者进行评测。指定测试数据集路径时该参数必传'
    )
    return parser.parse_args()


#Agent 调用薄函数
def call_agent(agent, test_case_csv: dict):
    """agent调用入口，将agent响应并入原有字典中"""
    logger.info(f'call_agent,qurey:{test_case_csv['query']}')
    answer_raw, answer_summary, res_time = agent.call_agent(test_case_csv['query'])
    test_case_csv['agent_response'] = answer_summary
    test_case_csv['res_time(s)'] = res_time
    logger.debug(answer_raw)
    logger.info(f'qurey:{test_case_csv['query']}请求完毕')


def make_llm_case(test_case_csv: dict):
    """llm_case组装方法，组装完成的用例重新放回字典中

    agent_response 缺失时（call_agent 失败）跳过该用例，写入错误标记。
    """
    if 'agent_response' not in test_case_csv:
        logger.debug(
            f"[跳过] query='{test_case_csv.get('query', '?')[:60]}' "
            f"agent_response 缺失，无法组装 LLMTestCase"
        )
        test_case_csv['is_success'] = False
        test_case_csv['evaluate_error'] = 'Agent调用失败，无agent_response'
        return

    llm_test_case = LLMTestCase(
        input=test_case_csv['query'],
        expected_output=test_case_csv.get('expected_behavior', ''),
        context=[test_case_csv.get('forbidden_behavior', '')],
        retrieval_context=[test_case_csv['agent_response']]
    )
    test_case_csv['llm_test_case'] = llm_test_case


#评测流水线
class EvaluationPipeline:
    """评测流水线，按阶段串联：准备 → Agent调用 → 评测 → 报告"""

    def __init__(self, csv_path: str | None, metrics: list[str] | None):
        self.conf = ConfigReader.get_instance()
        self.csv_path = csv_path
        self.metrics = metrics

    def run(self):
        test_cases_list = self._prepare()
        self._invoke_agents(test_cases_list)
        base_path = self._evaluate(test_cases_list)
        self._report(base_path)

    #阶段1: 用例准备
    def _prepare(self) -> list[dict]:
        return make_test_case_list(self.csv_path, self.metrics)

    #阶段2: Agent 并发调用 + 组装 LLM 用例
    def _invoke_agents(self, test_cases_list: list[dict]):
        """
        :param test_cases_list: [{csv: [dict], case_name: str, metrics: [str]}, ...]
        """
        agent = create_agent()
        # 每个csv表中的每一行
        all_cases = [
            case for cases in test_cases_list for case in cases['csv']
        ]

        max_worker = self.conf.get("agents.http_agent.call_agent_th_max")
        submit_delay = self.conf.get("agents.http_agent.submit_delay", 0)

        run_in_thread_pool(
            partial(call_agent, agent), all_cases,
            max_workers=max_worker, submit_delay=submit_delay,
            task_name="call_agent"
        )
        run_in_thread_pool(
            make_llm_case, all_cases,
            max_workers=max_worker, task_name='make_llm_test_case'
        )

        # 阶段统计
        ok = sum(1 for c in all_cases if 'llm_test_case' in c)
        ng = len(all_cases) - ok
        if ng:
            logger.info(f"[阶段2] Agent调用/用例组装: 成功 {ok}, 失败 {ng} (共 {len(all_cases)})")
        else:
            logger.info(f"[阶段2] Agent调用/用例组装: 全部成功 ({len(all_cases)} 条)")

    #阶段3: 评测执行 + 结果写入
    def _evaluate(self, test_cases_list: list[dict]) -> str:
        save_path = self.conf.get("result.save_path")
        base_path = mkdir_with_timestamp(save_path)

        writer = AsyncResultWriter(base_path)
        writer.start()

        for test_case in test_cases_list:
            run_evaluate(test_case)
            run_multimodel_reevaluate(test_case)
            writer.submit(test_case, test_case['case_name'])

        writer.wait_and_stop()
        print(writer.get_stats())
        return base_path

    #阶段4: 报告生成
    def _report(self, base_path: str):
        # 提取失败用例
        extract_bad_cases(base_path)

        report_path = os.path.join(base_path, 'test_report.md')
        md_writer = MarkdownWriter(report_path)

        for file in os.listdir(base_path):
            if file.endswith('.csv') and 'result_outputs_' in file:
                cr = CollectionResult(os.path.join(base_path, file))
                md_writer.write_report(
                    file, cr.task_success_stats(), cr.res_time()
                )


#入口
if __name__ == "__main__":
    params = parse_args()
    EvaluationPipeline(params.csv_path, params.metrics).run()
