# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

from deepeval.metrics import ContextualRecallMetric, GEval 
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
    """评测指标类，负责创建和管理评测指标实例"""
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
    

conf_reader = ConfigReader.get_instance()
# metric创建，单例
createmetrics = CreateMetrics()

# 自定义反向验证指标，测试数据集中的negative_criteria字段
# 可以作为幻觉评测
reverse_validation_thresholds = conf_reader.get('judge_thresholds.reverse_validation', 0.7)
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

# 鲁棒性评测指标


# ContextualRecallMetric metics 
contextual_recall_threshold = conf_reader.get('judge_thresholds.contextual_recall', 0.7)
contextual_recall_metric = createmetrics.create_contextual_recall_metric(threshold=contextual_recall_threshold)


if __name__ == "__main__":

    from deepeval import evaluate

    test_case = LLMTestCase(
        input="写 SQL 查询 2026-03 的数据，时间条件如何写更符合规则？",
        context=["使用 LIKE 过滤日期 | 对时间字段做函数包装后过滤"],
        retrieval_context=["### SQL 时间条件书写规则\n\n根据 DSP SQL 规则，在为 `2026-03` 这样的月份编写时间条件时，应遵循以下原则以确保查询正确性和性能：\n\n**核心原则**：优先对原始时间字段进行范围过滤，避免使用函数包装或 `LIKE`。\n\n#### **推荐写法**\n- **使用 `BETWEEN`**：当过滤整个月份时，这是最直观的写法。\n  ```sql\n  report_date BETWEEN '2026-03-01' AND '2026-03-31'\n  ```\n- **使用范围比较**：对于包含时分秒的字段，使用 `[起始, 结束)` 的左闭右开区间更精确。\n  ```sql\n  create_time >= '2026-03-01 00:00:00' AND create_time < '2026-04-01 00:00:00'\n  ```\n\n#### **应避免的写法**\n- **不要使用 `LIKE`**：这会导致性能问题且不精确。\n  - `report_date LIKE '2026-03%'`\n- **不要在过滤列上使用函数**：这会使索引失效。\n  - `DATE(create_time) >= '2026-03-01' AND DATE(create_time) <= '2026-03-31'`\n  - `SUBSTR(report_hour, 1, 7) = '2026-03'`\n\n**补充说明**：\n- 如果表有时间分区字段，应优先使用该字段进行过滤。\n- 所有时间均按 UTC+0 存储，只有在用户明确提供时区时才进行 `DATE_ADD` 换算。"]
    )

    test_case2 = LLMTestCase(
        input="用户说要查 Google 渠道的域名表现，能直接把 affiliate 当成 domain 吗？",
        context=["直接把渠道名等同于域名 | 混淆 first_ssp 与 affiliate"],
        retrieval_context=["### 简报：`affiliate`与`domain`字段在数据分析中的区别\n\n根据 `pac_dsp_response` 表结构文档，**不可以**将 `affiliate` 字段直接等同于 `domain` 字段来分析渠道表现。这两个字段代表了广告流量在不同层级的归属。\n\n#### 核心结论\n- **`affiliate` 代表渠道**: `affiliate_name` 字段指代的是广告流量交易所，即流量的来源**平台或渠道**，例如 'Google'、'OPPO_TecdoDSP'、'vivo'。\n- **`domain` 代表媒体**: `domain` 字段指代的是流量来源的**具体媒体**，即广告实际展示的网站域名。对于App内流量，对应的字段是 `bundle_id`（应用包名）。\n\n#### 字段定义解析\n\n| 字段名 | 描述 | 作用 |\n| :--- | :--- | :--- |\n| **`affiliate_name`** | 广告流量交易所的名称 | 用于区分大的流量渠道，如 Google、Vivo 等。 |\n| **`domain`** | 当前流量来源媒体（bundle）的域名 | 用于识别Web流量下，广告展示的具体网站。 |\n| **`bundle_id`** | 流量来源媒体的唯一标识符 | 用于识别App流量下，广告展示的具体应用。 |\n| **`traffic_type`** | 流量的来源类型 | 关键区分字段：`1` 代表App内流量，`2` 代表Web流量。 |\n\n#### 正确的分析方法\n要查询 Google 渠道下不同域名的表现，应遵循以下步骤：\n1.  **筛选渠道**: 使用 `WHERE affiliate_name LIKE '%Google%'`（或具体Google渠道的准确名称）来圈定 Google 渠道的流量。\n2.  **区分流量类型**:\n    *   对于 **Web 流量**（`traffic_type = 2`），使用 `domain` 字段进行分组（`GROUP BY domain`），以分析各个网站域名的表现。\n    *   对于 **App 流量**（`traffic_type = 1`），应使用 `bundle_id` 字段进行分组，以分析各个应用的表现。\n\n因此，`affiliate` 是分析的起点（渠道），而 `domain` 或 `bundle_id` 则是更深层次的分析维度（媒体）。"]
    )

    test_case3 = LLMTestCase(
        input="DSP 的广告投放五层级是什么？",
        context=["层级顺序错误 | 遗漏任一层级"],
        retrieval_context=["### DSP 广告投放五层级结构\n\n根据提供的业务背景资料，DSP 的广告投放采用自上而下的五层级结构：\n\n1.  **广告主 (adv)**：最高层级，代表广告客户。\n2.  **活动 (campaign)**：用于设定推广目标、日预算和目标平台（如 app/web）。\n3.  **广告组 (ad_group)**：负责具体的出价策略、定向条件、预算控制和优化目标。\n4.  **广告 (ad)**：创意的投放单元，关联具体的素材。\n5.  **素材 (creative)**：广告展示的最终内容，如图片、视频、图标等，存储于素材库。"]
    )

    test_case4 = LLMTestCase(
        input="dsp_report 和 brainx_report 内容一样，查哪个都行对吧？",
        context=["明确纠正：不是查哪个都行 | dsp_report：DSP 内部细维度分析 | brainx_report：客户视角天级汇总 | 应根据场景选表"],
        retrieval_context=["根据提供的文档，这种说法是**不正确**的。`dsp_report` 和 `brainx_report` 的用途不同，应根据具体查询场景选择。\n\n### 默认选表方向\n\n文档 `warmup.md` 和 `warmup.backup-2026-04-02.md` 中明确指出了两张表的区别和优先使用场景：\n\n*   **`default_catalog.ads_brainx.brainx_report`**:\n    *   **客户视角**、标准效果报表、天级分析时优先使用。\n\n*   **`default_catalog.ads.dsp_report`**:\n    *   **DSP 内部视角**、细维度分析、复杂排查时优先使用。\n\n因此，您应当根据分析的目的（是站在客户视角看标准报表，还是在 DSP 内部进行深度排查）来选择合适的表，而不是认为它们可以随意替换。"]
    )

    test_case5 = LLMTestCase(
        input="用户问某广搞主昨天花废多少，未特别说明口经。默认应该使用哪个字段？",
        expected_output="能识别'广搞主'为'广告主' | 能识别'花废'为'花费' | 能识别'口经'为'口径' | 默认口径是平台消耗 | 字段应为 price_total / 1000",
        retrieval_context=["### 查询广告主花费的默认字段\n\n当查询广告主昨日花费且未特别说明口径时，应默认使用 `adv_spend_not_rt` 字段。\n\n- **推荐字段**: `adv_spend_not_rt`\n  - **定义**: 该字段是当前正在使用的“广告主消耗”，即广告主支付给DSP平台的广告花费。\n- **废弃字段**: `adv_spend`\n  - **状态**: 此字段已被标记为“废弃”，为确保数据准确性，不应再使用。\n\n此结论基于知识库文档 `表结构: default_catalog.ads_brainx.brainx_report` 的描述。"]
    )   

    test = "编造具体创建日期 | 推测创建时间 | 假装查询了元数据"
    test_case6 = LLMTestCase(
        input="default_catalog.ads.dsp_report 表是什么时候创建的？",
        expected_output="明确说明知识库没有表创建时间信息 | 可以说明表的用途和结构 | 不应编造创建时间",
        context=test.split('|'),
        retrieval_context=["### `default_catalog.ads.dsp_report` 表创建时间分析\n\n根据现有资料，无法直接确定 `default_catalog.ads.dsp_report` 表本身的确切创建时间。\n\n不过，文档提供了与表中**数据创建时间**相关的字段信息：\n\n*   **`create_time`**:\n    *   **类型**: `varchar(20)`\n    *   **角色**: 表的分区键\n    *   **描述**: 代表表中数据的创建时间，格式为 `'YYYY-MM-DD_HH'`。\n\n*   **`create_date`**:\n    *   **类型**: `datetime`\n    *   **描述**: 代表数据的创建日期，格式为 `'YYYY-MM-DD HH:MM:SS'`。\n\n**结论**：已有的信息关注的是表中**行数据**的创建时间，而非表结构的创建时间。"]
    )   

    # evaluate([test_case, test_case2, test_case3], [reverse_validation_metric])
    # evaluate([test_case5], [reverse_validation_metric])
    # evaluate([test_case5], [contextual_recall_metric])
    evaluate([test_case6], [reverse_validation_metric, contextual_recall_metric])
    # for tr in resualt.test_results:
    #     pass
    # print(resualt)