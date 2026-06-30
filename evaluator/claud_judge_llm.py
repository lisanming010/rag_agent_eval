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

    ac_output = """
        光伏电站按是否与公共电网连接，可分为**独立光伏电站（离网系统）**、**并网光伏电站**以及**混合型光伏电站（并离网互补）**。具体如下：

        | 类型 | 定义 | 特点 |
        |------|------|------|
        | **独立光伏电站（离网系统）** | 不接入公共电网，自成发电-储能-用电闭环。 | 必须配备蓄电池储能，用于夜间或阴雨天供电。适用于无电网覆盖的偏远地区（如海岛、山区、边防哨所）。典型容量较小（几百瓦~几十千瓦），组件阵列与蓄电池组通过控制器充放电，逆变器仅需输出稳定交流电（正弦波或修正波）。 |
        | **并网光伏电站** | 直接接入公共电网，所发电力优先自用或全额上网。 | 可分“全额上网”和“自发自用、余电上网”两种消纳模式。不需要储能（但部分可配储能以参与调峰）。容量范围宽（从户用几千瓦到大型地面数百兆瓦）。投资成本较低，系统简单，是目前主流形式。 |
        | **混合型光伏电站（并离网互补）** | 同时具备并网和接入蓄电池的功能，支持并网模式，也可在电网停电时切换为离网独立供电。 | 采用双向逆变器（又称储能逆变器或混合逆变器），电池充放电可灵活调度。适用于对供电可靠性要求较高的用户（如医院、数据中心、有应急备电需求的工厂）。成本介于离网和纯并网之间，在配电网薄弱地区增多。 |

        > **运维提醒**：  
        > - 并网电站检修前，必须断开并网开关，确认电网侧已停电（可通过验电器判断），并严格按照“停电、验电、挂地线、挂牌”流程操作。  
        > - 独立电站的蓄电池区域需保持通风、干燥，防止氢气积聚（铅酸电池充电时可能析出氢气和氧气），避免明火和电气火花。  
        > - 混合型电站的储能逆变器在进行离网切换测试时，应确保负载侧已断开非重要负荷，防止瞬间过载损坏逆变器。

        ---
        >**来源**：大模型通用知识
    """

    test_case = LLMTestCase(
        input="用户问下单量应该查什么字段？",
        expected_output="应回答两类：（1）离网光伏电站；（2）并网光伏电站。",
        context=[],
        retrieval_context=[ac_output]
    )

    claude_judge = ClaudJudgeLLM()
    metric = claude_judge.create_contextual_recall_metric(threshold=0.7)
    evaluate([test_case], [metric])