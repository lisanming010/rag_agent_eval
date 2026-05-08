# rga_agent_eval

一个基于 `deepeval` 的 Agent 评测脚本项目,支持异步并发评测和结果统计分析。

## 项目结构

```text
.
├── agents/                        # Agent 注册目录
│   └── http_agent.py              # HTTP 方式调用待测 Agent
├── evaluator/                     # LLM-as-Judge 注册目录
│   ├── claud_judge_llm.py         # Claude Judge 模型封装
│   ├── deepeval_patch.py          # deepeval/anthropic 兼容性 patch
│   └── metrics.py                 # 自定义评测指标注册与构造
├── result/                        # 评测结果输出目录                 
├── test_suite/                    # 测试数据集存放目录
│   ├── test_cases_base.csv        # 基础测试集
│   ├── test_cases_hallucination.csv  # 幻觉测试集
│   ├── test_cases_inference.csv   # 推理测试集
│   ├── test_cases_robustness.csv  # 鲁棒性测试集
│   └── *.json                     # 对应的 JSON 数据
├── tool/                          # 工具模块
│   ├── __init__.py                # 工具模块导出
│   ├── async_result_writer.py     # 异步结果写入器
│   ├── collection_result.py       # 结果统计分析工具
│   ├── config_reader.py           # 配置读取工具(单例)
│   ├── markdown_writer.py         # 结果md输出工具
│   ├── csv_reader.py              # CSV 读取工具
│   └── csv_writer.py              # CSV 写入工具
├── .env.example                   # 环境变量示例
├── config.yaml.example            # 配置文件示例
├── main.py                        # 主执行入口
├── Pipfile                        # Python 依赖定义
├── requirements.txt               # pip 依赖列表
└── README.md                      # 项目文档
```

## 核心流程

主流程位于 [main.py](main.py),执行过程如下:

1. **读取配置**: 解析命令行参数和 [config.yaml](config.yaml) 配置
2. **加载测试集**: 从指定路径或默认目录加载 CSV 测试集
3. **并发调用 Agent**: 使用线程池并发调用待测 Agent,获取回答
4. **组装测试用例**: 将测试样本组装成 `deepeval` 的 `LLMTestCase`
5. **执行评测**: 根据配置的指标执行异步并发评测
6. **异步写入结果**: 评测完成后立即异步写入 CSV,不阻塞下一轮测试
7. **统计分析**: 所有测试完成后,统计各指标的成功率并输出报告

## 主要模块说明

### 1. 入口脚本 ([main.py](main.py))

- 定义命令行参数 `--csv_path`、`--metrics`
- 支持单个 CSV 文件评测或批量评测
- 使用线程池并发执行 Agent 调用
- 调用 `deepeval.evaluate()` 执行评测
- 使用 `AsyncResultWriter` 异步写入结果

### 2. Agent 调用层 ([agents/http_agent.py](agents/http_agent.py))

通过 HTTP POST 请求调用外部 Agent 接口:

```json
{
  "query": "用户问题",
  "mode": "FAST"
}
```

从响应中提取 `data.summary` 作为评测使用的回答文本。

### 3. 评测层

#### [evaluator/metrics.py](evaluator/metrics.py)

注册了两个核心评测指标，后续新的评测指标也应当在此注册:

- **`reverse_validation`**: 基于 `GEval` 自定义,反向校验,检查回答是否违反 `negative_criteria`(禁止事项)
- **`contextual_recall`**: 基于 `ContextualRecallMetric`,正向校验,评估回答与预期结果的一致性

#### [evaluator/claud_judge_llm.py](evaluator/claud_judge_llm.py)

- 封装 Judge 模型(默认使用 Claude Opus 4.7)
- 模型参数来自 [config.yaml](config.yaml)
- 认证信息来自 `.env` 文件

#### [evaluator/deepeval_patch.py](evaluator/deepeval_patch.py)

对 `deepeval` 使用 Anthropic SDK 时的兼容性问题进行 monkey patch:
- 移除不兼容的 `thinking` 参数
- 跳过 `ThinkingBlock`,提取真实文本
- 加强 JSON 解析与修复能力

### 4. 工具层

#### [tool/async_result_writer.py](tool/async_result_writer.py)

异步结果写入器,采用生产者-消费者模式:
- 主线程完成评测后立即提交结果到队列
- 独立的写入线程从队列取出结果并写入文件
- 测试和写入并行进行,提高整体效率

#### [tool/collection_result.py](tool/collection_result.py)

结果统计分析工具:
- 读取评测结果 CSV
- 计算总体成功率和各 metric 的通过率
- 支持按 metric 分组统计

#### [tool/config_reader.py](tool/config_reader.py)

- 单例配置读取器
- 支持通过 `a.b.c` 形式读取嵌套配置

#### [tool/csv_reader.py](tool/csv_reader.py) / [tool/csv_writer.py](tool/csv_writer.py)

- CSV 文件读写工具
- 支持字典列表格式
- 自动过滤不可序列化字段

## 环境要求

- Python 3.12+
- Pipenv 或 pip

主要依赖:
- `deepeval` - LLM 评测框架
- `anthropic` - Claude API SDK
- `requests` / `aiohttp` - HTTP 客户端
- `pyyaml` - YAML 配置解析
- `python-dotenv` - 环境变量管理
- `json-repair` - JSON 修复工具

## 安装与初始化

### 1. 安装依赖

使用 Pipenv:
```bash
pipenv install
```

或使用 pip:
```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

参考 [.env.example](.env.example) 创建 `.env` 文件:

```env
ANTHROPIC_BASE_URL="your-anthropic-base-url"
ANTHROPIC_AUTH_TOKEN="your-anthropic-api-key"
```

### 3. 配置项目参数

参考 [config.yaml.example](config.yaml.example) 编辑 [config.yaml](config.yaml),重点配置:

- `agents.http_agent.endpoint`: 待测 Agent 接口地址
- `agents.http_agent.mode`: Agent 调用模式(FAST/DEEP)
- `judge_llm.anthropic.model`: 评测用 Judge 模型（anthropic）
- `dataset.default_dataset_path`: 默认测试集路径
- `evluate.max_concurrent`: 评测并发数
- `evluate.throttle_value`: 请求节流间隔(秒)

## 使用方式

### 使用默认测试集执行

```bash
pipenv run python main.py
```

### 指定测试集与指标执行

```bash
pipenv run python main.py \
  --csv_path test_suite/test_cases_base.csv \
  --metrics reverse_validation contextual_recall
```

### 参数说明

- `--csv_path` / `-cp`
  - 指定单个 CSV 测试文件
  - 不传时使用配置中的默认路径

- `--metrics` / `-m`
  - 可选值:`reverse_validation`、`contextual_recall`
  - 指定了 `--csv_path` 时,该参数必传
  - 新增 metric 需要在 [evaluator/metrics.py](evaluator/metrics.py) 注册并在 main.py 中完成映射

## 配置说明

[config.yaml](config.yaml) 包含以下主要配置块:

### Agent 配置

```yaml
agents:
  type: http
  http_agent:
    endpoint: http://your-agent-endpoint/api/v1/search
    mode: FAST
    call_agent_th_max: 8  # Agent 调用最大并发数
```

### Judge 模型配置

```yaml
judge_llm:
  anthropic:
    model: claude-opus-4-7
    temperature: 0
    max_token: 4096
```

### 评测执行配置

```yaml
evluate:
  run_async: True          # 启用异步并发评测
  max_concurrent: 10       # 最大并发数
  throttle_value: 0        # 请求间隔(秒),建议设置为 1-2
```

**重要**: `throttle_value: 0` 可能导致触发 API 速率限制,建议设置为 1-2 秒。

### 数据集配置

```yaml
dataset:
  default_dataset_path: test_suite/  # 测试集目录或具体文件
  
  # 指定具体 CSV 文件时使用的 metrics
  metrics_if_specify_csv:
    - reverse_validation
    - contextual_recall
  
  # 自动匹配 test_cases_xxx.csv 时的 metrics 映射
  dataset_metrics_map:
    base:
      - reverse_validation
      - contextual_recall
    hallucination:
      - reverse_validation
      - contextual_recall
```

### 结果保存配置

```yaml
result:
  save_path: result/  # 结果输出目录
```

## 测试集格式

测试集命名规则:`test_cases_*.csv`,必须以 `test_cases_` 开头且为 CSV 格式。

### 单轮对话测试集

至少包含以下字段:

- `id`: 测试用例唯一标识
- `domain`: 测试领域
- `dimension`: 测试维度
- `difficulty`: 难度等级
- `question`: 发送给 Agent 的问题
- `expected_answer`: 期望回答要点
- `source_docs`: 样本依据来源
- `negative_criteria`: 禁止事项,用 `|` 分隔

示例见 [test_suite/test_cases_base.csv](test_suite/test_cases_base.csv)。

### 多轮对话测试集

包含以下字段:

- `id`: 测试用例唯一标识
- `multiturn_type`: 多轮对话类型
- `domain`: 测试领域
- `difficulty`: 难度等级
- `total_turns`: 总轮次数
- `turn1_expected`, `turn1_message`, `turn1_speaker`: 第 1 轮的期望、消息、说话者
- `turn2_expected`, `turn2_message`, `turn2_speaker`: 第 2 轮...
- `context_dependencies`: 上下文依赖关系
- `negative_criteria`: 禁止事项

## 输出结果

结果保存在 `result/` 目录下,按时间戳命名的子目录中:

```
result/
  └── 0507123045/
      ├── result_output_base.csv
      ├── result_output_hallucination.csv
      ├── result_output_robustness.csv
      └── test_report.md                  # 测试结果汇总，当前有各指标完成率和任务执行时间相关统计
```

### 结果字段

输出 CSV 包含:

- **原始测试字段**: 所有输入字段
- **Agent 响应**: `agent_response`
- **总体结果**: `is_success`
- **各指标结果**:
  - `{metric}_is_success`: 该指标是否通过
  - `{metric}_score`: 该指标得分(0-1)
  - `{metric}_threshold`: 该指标阈值
  - `{metric}_reason`: 该指标评判理由

## TODO

- [ ] 召回结果验证包括关联的MRR、Recall@K、Precision@K指标，需要能够拿到相关召回文档列表
- [x] 生成 HTML/Markdown 格式的详细报告
