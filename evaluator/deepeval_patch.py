"""
补全了 monkey patch：

继续移除 _build_client 里的 thinking 参数
patch 了 generate / a_generate，从 message.content 中跳过 ThinkingBlock，提取真正的 TextBlock.text
增强 JSON 解析，支持 markdown code block 和前后缀文本
补全异步评价的进度收尾，避免异常/跳过/取消的用例行残留
create by:cc
"""

import re
import json
import logging
from functools import wraps
from json_repair import repair_json
from deepeval.models.llms import anthropic_model
from deepeval.models.llms.anthropic_model import require_secret_api_key
from deepeval.errors import DeepEvalError


_PATCHED = False
logger = logging.getLogger(__name__)


def patch_evaluation_progress():
    """保证每个异步 metric 结束时恰好推进一次用例子进度。

    DeepEval 的 safe_a_measure 在异常/取消/跳过分支不更新进度，导致
    Rich 子任务无法完成并移除。保留原有评价和异常处理，只把其进度参数
    置空，将进度更新统一到 finally；多指标并发时也不会重复计数。
    """
    from deepeval.metrics import indicator

    original = indicator.safe_a_measure
    if getattr(original, "_rga_progress_cleanup_patched", False):
        return

    @wraps(original)
    async def patched_safe_a_measure(
        metric, tc, ignore_errors, skip_on_missing_params,
        progress=None, pbar_eval_id=None, _in_component=False,
    ):
        try:
            return await original(
                metric, tc, ignore_errors, skip_on_missing_params,
                progress=None,
                pbar_eval_id=None,
                _in_component=_in_component,
            )
        finally:
            try:
                # update_pbar 对关闭展示或已删除的任务是 no-op；完成时自动移除。
                indicator.update_pbar(progress, pbar_eval_id)
            except Exception:
                # 展示异常不能覆盖模型异常，也不能让正常评分变成失败。
                logger.warning("DeepEval 子进度清理失败", exc_info=True)

    patched_safe_a_measure._rga_progress_cleanup_patched = True
    indicator.safe_a_measure = patched_safe_a_measure


def _extract_text(message):
    """从 Anthropic 响应的 content 列表中提取 TextBlock 的 text，跳过 ThinkingBlock"""
    for block in message.content:
        if hasattr(block, "text"):
            return block.text
    return message.content[0].text


def _append_json_constraint(prompt: str) -> str:
    """在 prompt 末尾追加 JSON 格式约束"""
    return (
        prompt
        + "\n\nIMPORTANT: Return valid JSON only. Do not wrap in markdown code blocks."
        + " Escape all double quotes inside JSON string values with backslashes."
    )


def _robust_json_parse(text: str, message=None) -> dict:
    """
    增强的 JSON 解析，支持：
    1. markdown code block: ```json ... ```
    2. 前后缀文本
    3. 尾部逗号清理
    """
    # 尝试1: 去除 markdown code block
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1)

    # 尝试2: 提取 { ... } 之间的内容
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        json_str = text[start:end]
    else:
        json_str = text

    # 清理尾部逗号
    json_str = re.sub(r',\s*([\]}])', r'\1', json_str)

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON 解析失败")
        if message:
            stop_reason = getattr(message, 'stop_reason', 'unknown')
            output_tokens = getattr(message.usage, 'output_tokens', 'unknown') if hasattr(message, 'usage') else 'unknown'
            print(f"[DEBUG] stop_reason: {stop_reason}")
            print(f"[DEBUG] output_tokens: {output_tokens}")
        print(f"[DEBUG] text_len: {len(text)}")
        print(f"[DEBUG] 原始文本前 500 字符:")
        print(text[:500])
        print(f"[DEBUG] 原始文本末尾 300 字符:")
        print(text[-300:])
        raise DeepEvalError(
            f"Evaluation LLM outputted an invalid JSON: {str(e)}\n"
            f"Text preview: {text[:200]}"
        )


def patch_anthropic_model():
    """
    修复 deepeval AnthropicModel 与当前 anthropic SDK 的兼容性问题:
    1. 初始化时移除不支持的 thinking 参数
    2. generate/a_generate 中跳过 ThinkingBlock，取 TextBlock
    """
    global _PATCHED
    if _PATCHED:
        return

    AnthropicModel = anthropic_model.AnthropicModel

    # patch 1: _build_client 移除 thinking 参数
    def patched_build_client(self, cls):
        api_key = require_secret_api_key(
            self.api_key,
            provider_label="Anthropic",
            env_var_name="ANTHROPIC_API_KEY",
            param_hint="`api_key` to AnthropicModel(...)",
        )
        kw = dict(api_key=api_key, **self._client_kwargs())
        kw.pop("thinking", None)
        try:
            return cls(**kw)
        except TypeError as e:
            if "max_retries" in str(e):
                kw.pop("max_retries", None)
                return cls(**kw)
            raise

    # patch 2: generate 跳过 ThinkingBlock
    def patched_generate(self, prompt, schema=None):
        from deepeval.models.llms.anthropic_model import (
            check_if_multimodal, convert_to_multi_modal_array,
        )

        # 当需要结构化输出时，追加 JSON 约束
        if schema is not None:
            prompt = _append_json_constraint(prompt)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        chat_model = self.load_model()
        message = chat_model.messages.create(
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": content}],
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        text = _extract_text(message)
        if schema is None:
            return text, cost
        else:
            try:
                json_output = _robust_json_parse(text, message)
            except DeepEvalError:
                print("[DEBUG] 尝试用 json_repair 修复后重试")
                repaired = repair_json(text)
                json_output = json.loads(repaired)
            return schema.model_validate(json_output), cost

    # patch 3: a_generate 跳过 ThinkingBlock
    async def patched_a_generate(self, prompt, schema=None):
        from deepeval.models.llms.anthropic_model import (
            check_if_multimodal, convert_to_multi_modal_array,
        )

        # 当需要结构化输出时，追加 JSON 约束
        if schema is not None:
            prompt = _append_json_constraint(prompt)

        if check_if_multimodal(prompt):
            prompt = convert_to_multi_modal_array(input=prompt)
            content = self.generate_content(prompt)
        else:
            content = [{"type": "text", "text": prompt}]

        chat_model = self.load_model(async_mode=True)
        message = await chat_model.messages.create(
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": content}],
            model=self.name,
            temperature=self.temperature,
            **self.generation_kwargs,
        )
        cost = self.calculate_cost(
            message.usage.input_tokens, message.usage.output_tokens
        )
        text = _extract_text(message)
        if schema is None:
            return text, cost
        else:
            try:
                json_output = _robust_json_parse(text, message)
            except DeepEvalError:
                print("[DEBUG] 尝试用 json_repair 修复后重试")
                repaired = repair_json(text)
                json_output = json.loads(repaired)
            return schema.model_validate(json_output), cost

    AnthropicModel._build_client = patched_build_client
    AnthropicModel.generate = patched_generate
    AnthropicModel.a_generate = patched_a_generate
    _PATCHED = True
