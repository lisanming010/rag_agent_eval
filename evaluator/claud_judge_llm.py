import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval.models import AnthropicModel, GPTModel
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric
from dotenv import load_dotenv
from tool.config_reader import ConfigReader
from evaluator.deepeval_patch import patch_anthropic_model
import os

load_dotenv()
patch_anthropic_model()

class ClaudJudgeLLM:
    """实例化基于claude模型"""
    def __init__(self):
        self.base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        self.model = None

    def get_model(self):
        """懒加载模型实例"""
        configreader = ConfigReader.get_instance()
        model_name = configreader.get("judge_llm.anthropic.model")
        model_temperature = configreader.get("judge_llm.anthropic.temperature")
        model_max_token = configreader.get("judge_llm.anthropic.max_token")

        if self.model is None:
            self.model = AnthropicModel(
                model=model_name,
                base_url=self.base_url,
                api_key=self.auth_token,
                temperature=model_temperature,
                max_tokens=model_max_token,
            )
        return self.model
 
    def get_model_openai(self):
        """
        懒加载模型实例，使用OpenAI兼容格式调用,原生AnthropicModel可能会出现'Thinking block error'
        """
        configreader = ConfigReader.get_instance()
        model_name = configreader.get("judge_llm.anthropic.model")
        model_temperature = configreader.get("judge_llm.anthropic.temperature")

        if self.model is None:
            self.model = GPTModel(
                model=model_name,
                base_url=self.base_url + "/v1",
                api_key=self.auth_token,
                temperature=model_temperature,
                generation_kwargs={
                "response_format": {"type": "json_object"}
                }
            )
        return self.model

    def create_metric(self, threshold=0.7):
        """创建基于claude judge的评测指标"""
        model = self.get_model()
        return AnswerRelevancyMetric(model=model, threshold=threshold)
    
    def create_contextual_recall_metric(self, threshold=0.7):
        """创建基于claude judge的上下文召回评测指标"""
        model = self.get_model()
        return ContextualRecallMetric(model=model, threshold=threshold)
    
if __name__ == "__main__":
    from deepeval.test_case import LLMTestCase
    from deepeval import evaluate

    ac_output = "### \"下单量\"指标查询指南\n\n根据业务文档中的默认指标语义，当查询“下单量”或“订单数”时，应使用以下字段进行统计：\n\n*   **指标字段**: `event11_count`\n*   **计算方式**: `SUM(event11_count)`\n\n#### 关键点\n- **`warmup.md` 文档**明确指出，“下单量”的默认指标语义是 `SUM(event11_count)`。\n- **`business-context.md`** 提供了更广泛的业务背景，但对于此具体指标的定义，`warmup` 文件中的启发式规则提供了最直接的答案。"

    test_case = LLMTestCase(
        input="用户问下单量应该查什么字段？",
        expected_output="下单量优先看 SUM(event11_count)",
        retrieval_context=[ac_output]
    )

    claude_judge = ClaudJudgeLLM()
    metric = claude_judge.create_contextual_recall_metric(threshold=0.7)
    evaluate([test_case], [metric])