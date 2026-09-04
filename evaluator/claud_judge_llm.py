import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval.models import AnthropicModel, GPTModel
from deepeval.metrics import AnswerRelevancyMetric, ContextualRecallMetric
from dotenv import load_dotenv
from tool.config_reader import ConfigReader
from evaluator.deepeval_patch import patch_anthropic_model, patch_evaluation_progress
import os

load_dotenv()
patch_anthropic_model()
patch_evaluation_progress()

class ClaudJudgeLLM:
    """实例化基于claude模型，支持多模型实例缓存（按 model_name 隔离）"""

    def __init__(self):
        self.base_url = os.getenv("ANTHROPIC_BASE_URL")
        self.auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")
        self._models: dict[str, AnthropicModel] = {}
        self._openai_models: dict[str, GPTModel] = {}

    def get_model(self, model_name: str | None = None):
        """
        懒加载模型实例

        :param model_name: 可选指定模型名称，不传则使用配置文件默认 model
        """
        if model_name is None:
            model_name = ConfigReader.get_instance().get("judge_llm.anthropic.model")

        if model_name not in self._models:
            configreader = ConfigReader.get_instance()
            model_temperature = configreader.get("judge_llm.anthropic.temperature")
            model_max_token = configreader.get("judge_llm.anthropic.max_token")

            self._models[model_name] = AnthropicModel(
                model=model_name,
                base_url=self.base_url,
                api_key=self.auth_token,
                temperature=model_temperature,
                max_tokens=model_max_token,
            )
        return self._models[model_name]

    def get_model_openai(self, model_name: str | None = None):
        """
        懒加载模型实例，使用OpenAI兼容格式调用,原生AnthropicModel可能会出现'Thinking block error'

        :param model_name: 可选指定模型名称，不传则使用配置文件默认 model
        """
        if model_name is None:
            model_name = ConfigReader.get_instance().get("judge_llm.anthropic.model")

        if model_name not in self._openai_models:
            configreader = ConfigReader.get_instance()
            model_temperature = configreader.get("judge_llm.anthropic.temperature")
            model_max_token = configreader.get("judge_llm.anthropic.max_token")

            self._openai_models[model_name] = GPTModel(
                model=model_name,
                base_url=self.base_url + "/v1",
                api_key=self.auth_token,
                temperature=model_temperature,
                generation_kwargs={
                    "response_format": {"type": "json_object"},
                    "max_tokens": model_max_token,
                    "extra_body": {"enable_thinking": False}
                }
            )
        return self._openai_models[model_name]

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
Current Diagnostic Target: Inverter Type -> Manufacturer (Sungrow) -> Sungrow General Model -> Fault Code -> MPPT 1 reverse connection

---

### **Fault Name: MPPT 1 reverse connection**
**Fault Code:** `264`

**Fault Code 264 - MPPT 1 Reverse Connection**

**Possible Cause:**
The PV string connected to the MPPT 1 channel has been wired with reversed DC polarity.

**Action Steps:**
1. **Automatic Recovery:** This fault is **non-auto-recoverable**. It requires manual inspection, a complete power shutdown, and a physical reset of the wiring.
2. **Test DC Polarity:** Use a multimeter to verify the DC voltage polarity at the MPPT 1 input terminals.
3. **Safety Protocol:** Wait until the string current drops below 5A before opening the DC switch to correct the wiring polarity.
4. **Correct Wiring:** Open the DC switch, correct the reversed polarity, and then close the DC switch.
5. **Verify:** Confirm that the MPPT 1 reverse connection alarm has cleared from the inverter display.

---

**Guidance:**
This procedure is specific to the MPPT 1 input channel. If the fault code persists after correcting the polarity and resetting the system, the issue may be internal to the inverter's MPPT circuitry rather than the external PV wiring. To proceed, please confirm whether the DC voltage at the MPPT 1 terminals reads a positive value (e.g., +400V) or a negative value (e.g., -400V) relative to ground, as this will determine if the wiring correction was successful or if further hardware diagnostics are needed.

---
Hope this helps. You can now:
<suggest>learn more</suggest>
<suggest>save to case base</suggest>
    """

    test_case = LLMTestCase(
        input="用户问下单量应该查什么字段？",
        expected_output="""
Device Type (Inverter) → Manufacturer (Sungrow) → Model (General) → Fault Code → MPPT 1 reverse connection,Possible Cause:
PV string connected to MPPT1 channel wired with reversed DC polarity.

Action:
1. **Automatic Recovery**: **【Non-auto-recoverable】** (Requires manual inspection, power shutdown, and physical reset);
2. **Test dc**: Test DC voltage polarity at MPPT1 input terminals using a multimete;
3. **Safety Protocol**: **Wait until string current drops below;
4. **5a before**: 5A before opening DC switch to correct wiring polarity**;
5. **Close dc**: Close DC switch and verify MPPT1 reverse connection alarm clears;
""",
        context=[],
        retrieval_context=[ac_output]
    )

    # claude_judge = ClaudJudgeLLM().get_model('ZhipuAI/GLM-5.2')
    claude_judge = ClaudJudgeLLM().get_model_openai('Qwen/Qwen3.6-27B')
    metric = ContextualRecallMetric(model=claude_judge, threshold=0.7)
    # metric = claude_judge.create_contextual_recall_metric(threshold=0.7)
    evaluate([test_case], [metric])
