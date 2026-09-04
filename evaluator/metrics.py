# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from deepeval.metrics import ContextualRecallMetric, GEval, BaseMetric
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

from tool.config_reader import ConfigReader
from evaluator.claud_judge_llm import ClaudJudgeLLM

TESTCASE_PARAMS_MAP = {
    "input": LLMTestCaseParams.INPUT,
    "actual_output": LLMTestCaseParams.ACTUAL_OUTPUT,
    "expected_output": LLMTestCaseParams.EXPECTED_OUTPUT,
    "context": LLMTestCaseParams.CONTEXT,
    "retrieval_context": LLMTestCaseParams.RETRIEVAL_CONTEXT,
    "tools_called": LLMTestCaseParams.TOOLS_CALLED,
    "expected_tools": LLMTestCaseParams.EXPECTED_TOOLS,
    "mcp_servers": LLMTestCaseParams.MCP_SERVERS,
    "mcp_tools_called": LLMTestCaseParams.MCP_TOOLS_CALLED,
    "mcp_resources_called": LLMTestCaseParams.MCP_RESOURCES_CALLED,
    "mcp_prompts_called": LLMTestCaseParams.MCP_PROMPTS_CALLED,
}

class CreateMetrics:
    """评测指标类，负责创建和管理内置的或基于GEval的自定义评测指标实例"""

    def __init__(self, model_name: str | None = None):
        """
        :param model_name: 可选指定评测模型名称，不传则使用配置文件默认 model
        """
        # self.model = ClaudJudgeLLM().get_model(model_name)
        self.model = ClaudJudgeLLM().get_model_openai(model_name)

    def create_contextual_recall_metric(self, threshold=0.7) -> ContextualRecallMetric:
        """
        创建ContextualRecall metric，评价召回是否全面

        :params: threshold: 通过阈值
        """
        metric = ContextualRecallMetric(
            model=self.model,
            threshold=threshold
        )

        return metric
    
    def create_metric_base_geval(
            self,
            name: str,
            criteria: str,
            evaluation_params: list[str],
            evaluation_steps: list[str]|None = None,
            threshold=0.7,
            **kwargs
    )-> GEval:
        """
        自定义基于geval的metric

        :params: name: 指标名称，用于标识这个评估指标
        :params: criteria: 使用自然语言告诉LLM评估什么
        :params: evaluation_params: LLMTestCase中的参数对象
        :params: evaluation_steps: 具体评估步骤可为空
        :params: threshold: 通过阈值
        :params: kwargs: 其他参数, 其他GEval原生参数，可直接传入
        """

        # LLMTestCase参数转换与校验
        eval_params_list = []
        for param in evaluation_params:
            if param not in TESTCASE_PARAMS_MAP:
                raise ValueError(f"evaluation_params中参数 {param} 不合法，"
                                 f"必须是LLMTestCaseParams的参数: {list(TESTCASE_PARAMS_MAP.keys())} 之一")
            else:
                eval_params_list.append(TESTCASE_PARAMS_MAP[param])

        metric = GEval(
            name=name,
            model=self.model,
            criteria=criteria,
            evaluation_params=eval_params_list,
            evaluation_steps=evaluation_steps,
            threshold=threshold,
            **kwargs
        )

        return metric

class MRRMetric(BaseMetric):
    """自定义MRR指标评判类，非LLM"""

    def __init__(self, threshold: float=0.7, topk: int=0):
        self.threshold = threshold
        self.topk = topk

    def measure(self, test_case: LLMTestCase)->float:
        """
        计算MRR并评分

        可能预期内有多个文档均期望被召回，仅看期望召回文档列表中排名最靠前的结果
        如：expected = [a, b, c]，实际召回：[x, z, b, c, y]，那最终MRR以b的结果为准

        retrieval_context: 实际返回的候选列表（按相关性排序）
        expected_output: 期望命中的文档/答案
        """
        retrieved = test_case.retrieval_context or []
        # topk截断
        if self.topk > 0:
            retrieved = retrieved[:self.topk]

        # 避免切分出空值
        expected = [item.strip() for item in (test_case.expected_output or '').split('|') if item.strip()]

        self.score = 0.0
        first_recall_doc = ''
        for ex in expected:
            ex = ex.strip()
            for rank, item in enumerate(retrieved, start=1):
                if ex in item:
                    curr_score = 1 / rank
                    if curr_score > self.score:
                        self.score = curr_score
                        first_recall_doc = ex
                        break

        self.success = self.score >= self.threshold
        if self.score > 0:
            self.reason = f'第一个命中的文档是：{first_recall_doc}, 命中位置在：{int(1/self.score)}'
        else:
            self.reason = '未命中任何期望文档'
        return self.score
    
    async def a_measure(self, test_case: LLMTestCase)->float:
        return self.measure(test_case)
    
    def is_successful(self):
        return self.success
    
    @property
    def __name__(self):
        return "MRR"


class RetrievalKMetricBase(BaseMetric):
    def __init__(self, threshold: float = 0.7, topk: int = 0):
        self.threshold = threshold
        self.topk = topk

    def _prepare(self, test_case: LLMTestCase) -> tuple[list[str], list[str]]:
        retrieved = test_case.retrieval_context or []
        if self.topk > 0:
            retrieved = retrieved[:self.topk]

        expected = [item.strip() for item in (test_case.expected_output or '').split('|') if item.strip()]
        return retrieved, expected

    def _count_hits(self, retrieved: list[str], expected: list[str]) -> tuple[int, list[str]]:
        hit_count = 0
        hit_items = []
        for ex in expected:
            for item in retrieved:
                if ex in item:
                    hit_count += 1
                    hit_items.append(ex)
                    break
        return hit_count, hit_items

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self):
        return self.success


class RecallK(RetrievalKMetricBase):
    """自定义Recall@K评测指标，召回覆盖率，非LLM"""

    def measure(self, test_case: LLMTestCase) -> float:
        retrieved, expected = self._prepare(test_case)
        hit_count, hit_items = self._count_hits(retrieved, expected)

        self.score = hit_count / len(expected) if expected else 0.0
        self.success = self.score >= self.threshold
        if hit_items:
            self.reason = f'命中的期望项：{" | ".join(hit_items)}，Recall@K={self.score:.4f}'
        else:
            self.reason = '未命中任何期望项'
        return self.score

    @property
    def __name__(self):
        return "RecallK"


class PrecisionK(RetrievalKMetricBase):
    """自定义Precision@K评测指标，召回准确率，非LLM"""

    def measure(self, test_case: LLMTestCase) -> float:
        retrieved, expected = self._prepare(test_case)
        hit_count, hit_items = self._count_hits(retrieved, expected)

        self.score = hit_count / len(retrieved) if retrieved else 0.0
        self.success = self.score >= self.threshold
        if hit_items:
            self.reason = f'命中的期望项：{" | ".join(hit_items)}，Precision@K={self.score:.4f}'
        else:
            self.reason = '前K个结果中未命中任何期望项'
        return self.score

    @property
    def __name__(self):
        return "PrecisionK"


class DataQACapabilityMetric(BaseMetric):
    """校验 DataQA 意图路由是否正确 — 预期 capability_id 与实际返回的是否一致，非 LLM"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """
        expected_output: 预期 capability_id 字符串，如 "Station_power"
        retrieval_context[0]: 实际 capability_id 字符串
        """
        expected = (test_case.expected_output or '').strip()
        actual = ''
        if test_case.retrieval_context:
            actual = (test_case.retrieval_context[0] or '').strip()

        self.score = 1.0 if expected and actual and expected == actual else 0.0
        self.success = self.score >= self.threshold

        if not expected:
            self.reason = f'预期 capability_id 为空，实际={actual}'
        elif self.score == 1.0:
            self.reason = f'capability_id 匹配: {actual}'
        else:
            self.reason = f'capability_id 不匹配: 期望={expected}, 实际={actual}'
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "DataQA_Capability"


class DataQAParamsMetric(BaseMetric):
    """校验 DataQA 参数提取是否正确 — 预期参数集是否为实际参数集的子集，非 LLM"""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def measure(self, test_case: LLMTestCase) -> float:
        """
        expected_output: 预期参数字典的 JSON 字符串，如 '{"siteId":"Win Win"}'
        retrieval_context[0]: 实际参数字典的 JSON 字符串
        判定: expected ⊆ actual
        """
        expected_raw = (test_case.expected_output or '').strip()
        actual_raw = ''
        if test_case.retrieval_context:
            actual_raw = (test_case.retrieval_context[0] or '').strip()

        # 解析两个 JSON 字典
        expected_dict: dict = {}
        actual_dict: dict = {}
        try:
            expected_dict = json.loads(expected_raw) if expected_raw else {}
        except json.JSONDecodeError:
            self.score = 0.0
            self.success = False
            self.reason = f'expected_parameters JSON 解析失败: {expected_raw[:200]}'
            return 0.0
        try:
            actual_dict = json.loads(actual_raw) if actual_raw else {}
        except json.JSONDecodeError:
            self.score = 0.0
            self.success = False
            self.reason = f'actual_parameters JSON 解析失败: {actual_raw[:200]}'
            return 0.0

        if not expected_dict:
            self.score = 1.0
            self.success = True
            self.reason = '无预期参数，跳过校验'
            return 1.0

        # 子集比对: expected ⊆ actual
        matched: list[str] = []
        mismatched: list[str] = []
        for key, expected_val in expected_dict.items():
            actual_val = actual_dict.get(key)
            if actual_val is not None and str(actual_val) == str(expected_val):
                matched.append(f'{key}={expected_val}')
            else:
                mismatched.append(
                    f'{key}: 期望={expected_val}, 实际={actual_val}'
                )

        total = len(expected_dict)
        hit_count = len(matched)
        self.score = hit_count / total

        threshold_to_use = self.threshold
        # 当 threshold=1.0 时，要求全部匹配
        self.success = hit_count == total if threshold_to_use >= 1.0 else self.score >= threshold_to_use

        if mismatched:
            self.reason = (
                f'匹配: [{"; ".join(matched)}]; '
                f'不匹配: [{"; ".join(mismatched)}]'
            ) if matched else f'不匹配: [{"；".join(mismatched)}]'
        else:
            self.reason = f'全部匹配 ({total}/{total}): [{"; ".join(matched)}]'
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "DataQA_Params"


conf_reader = ConfigReader.get_instance()


def create_metrics_for_model(model_name: str | None, metric_names: list[str]) -> dict:
    """
    为指定模型创建 LLM 评测指标实例，非 LLM 类指标自动跳过。
    统一工厂，model1/model2/model3 均走此入口。

    :param model_name: 评测模型名称，传 None 则使用配置文件默认 model
    :param metric_names: 需要创建的指标名称列表
    :return: {metric_name: metric_instance}
    """
    factory = CreateMetrics(model_name=model_name)
    result: dict = {}

    for name in metric_names:
        if name == 'reverse_validation':
            threshold = conf_reader.get('metric_conf.reverse_validation.threshold', 0.7)
            result[name] = factory.create_metric_base_geval(
                name="reverse_validation_metric",
                criteria="retrieval_context中不应该包含context中的关键信息",
                evaluation_params=["input", "retrieval_context", "context"],
                evaluation_steps=[
                    "步骤一：retrieval_context是待评测的大模型的实际输出，context是禁止条例，不希望大模型输出中包含的内容",
                    "步骤二：若retrieval_context中有'来源：大模型通用知识'字段标识且仅有该字段，则直接认为该条输出合法，不再执行后续步骤也不再判断禁止条例，判断为满分后退出评价。其余'来源：xxx'只要不是'大模型通用知识'的都不适用于该步骤的豁免条件正常执行后续步骤判断。",
                    "步骤三：context中信息可能是肯定或否定的陈述，你应当理解肯定的陈述默认是缺省了'不应该'，如：'认为xxx'实际应该按照'不应该认为xxx'来理解",
                    "步骤四：执行比较，如果retrieval_context中体现了任一不该体现的信息则判定违禁，打分0分",
                    "步骤五：如果retrieval_context中也明确禁止了context中禁止的的操作例如：context中有：'不应该xxx'，在retrieval_context中也有'禁止xxx'或类似表述则不视为违禁，打分满分",
                    "步骤六：如果retrieval_context中没有体现禁止项则视为未违禁，比如：context中有：'不应该A'，retrieval_context中做了B、C则也同样视为不违禁，打分满分"
                ],
                threshold=threshold
            )
        elif name == 'contextual_recall':
            threshold = conf_reader.get('metric_conf.contextual_recall.threshold', 0.7)
            result[name] = factory.create_contextual_recall_metric(threshold=threshold)
    return result


# 默认模型（model1）LLM 指标单例，统一走工厂
_default_llm_metrics = create_metrics_for_model(None, ['reverse_validation', 'contextual_recall'])
reverse_validation_metric = _default_llm_metrics['reverse_validation']
contextual_recall_metric = _default_llm_metrics['contextual_recall']

# MRRmteric
mrr_threshold = conf_reader.get('metric_conf.mrr.threshold', 0.7)
mrr_topk = conf_reader.get('metric_conf.mrr.topk', 0)
mrr_metric = MRRMetric(threshold=mrr_threshold, topk=mrr_topk)

# recall@K
recall_k_threshold = conf_reader.get('metric_conf.recall_k.threshold', 0.8)
recall_k_topk = conf_reader.get('metric_conf.recall_k.topk', 5)
recallk_metric = RecallK(threshold=recall_k_threshold, topk=recall_k_topk)

# precision@k
precision_k_threshold = conf_reader.get('metric_conf.precision_k.threshold', 0.6)
precision_k_topk = conf_reader.get('metric_conf.precision_k.topk', 5)
precisionk_metric = PrecisionK(threshold=precision_k_threshold, topk=precision_k_topk)

# DataQA capability
dataqa_capability_threshold = conf_reader.get('metric_conf.dataqa_capability.threshold', 1.0)
dataqa_capability_metric = DataQACapabilityMetric(threshold=dataqa_capability_threshold)

# DataQA params
dataqa_params_threshold = conf_reader.get('metric_conf.dataqa_params.threshold', 1.0)
dataqa_params_metric = DataQAParamsMetric(threshold=dataqa_params_threshold)