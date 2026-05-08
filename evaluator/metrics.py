# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

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
    def __init__(self):
        self.model = ClaudJudgeLLM().get_model()

    def create_contextual_recall_metric(self, threshold=0.7) -> ContextualRecallMetric:
        """
        创建ContextualRecall metric

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
                if item in ex:
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

conf_reader = ConfigReader.get_instance()
# metric创建，单例
createmetrics = CreateMetrics()

# 自定义反向验证指标，测试数据集中的negative_criteria字段
# 可以作为幻觉评测
reverse_validation_thresholds = conf_reader.get('metric_conf.reverse_validation.threshold', 0.7)
reverse_validation_metric = createmetrics.create_metric_base_geval(
    name="reverse_validation_metric",
    criteria="retrieval_context中不应该包含context中的关键信息",
    evaluation_params=["input", "retrieval_context", "context"],
    evaluation_steps= [
        "context是禁止条例，是retrieval_context中不应该体现的内容或执行的操作",
        # "阅读context,其中的关键信息之间使用’｜’分割",
        "context中信息可能是肯定或否定的陈述，你应当理解肯定的陈述默认是缺省了’不应该’，如：’认为xxx’实际应该按照’不应该认为xxx’来理解",
        "理解context中的各关键信息",
        "执行比较，如果retrieval_context中体现了任一不该体现的信息则判定违禁，打分0分",
        "如果retrieval_context中也明确禁止了context中禁止的的操作例如：context中有：’不应该xxx’，在retrieval_context中也有’禁止xxx’或类似表述则不视为违禁，打分100分",
        "如果retrieval_context中没有体现禁止项则视为未违禁，比如：context中有：’不应该A’，retrieval_context中做了B、C则也同样视为不违禁，打分100分"
    ],
    threshold=reverse_validation_thresholds
)

# ContextualRecallMetric metics 
contextual_recall_threshold = conf_reader.get('metric_conf.contextual_recall.threshold', 0.7)
contextual_recall_metric = createmetrics.create_contextual_recall_metric(threshold=contextual_recall_threshold)

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