# rga_agent_eval

一个基于 `deepeval` 的 Agent 评测脚本项目

## 项目结构

```text
.
├── agents/												 # agent注册目录
│   └── http_agent.py              # HTTP 方式调用待测 Agent
├── evaluator/										 # LLM-as-Judger注册目录，在此处注册其余ai供应商
│   ├── claud_judge_llm.py         # Claude Judge 模型封装
│   ├── deepeval_patch.py          # 对 deepeval / anthropic 的兼容性 patch，规避think block text error
│   └── metrics.py                 # 自定义评测指标注册与构造，单例返回
├── result/
│   └── .gitkeep                   # 预留结果目录
├── test_suite/										 # 默认测试数据集存放目录
│   ├── test_cases_*.csv           # 多类测试集
│   └── test_cases_*.json          # 对应 JSON 数据
├── tool/
│   ├── config_reader.py           # 配置读取工具，单例调用
│   ├── csv_reader.py              # CSV 读取工具
│   └── csv_writer.py              # CSV 写入工具
├── .env.example                   # 环境变量示例
├── config.yaml                    # 项目配置
├── main.py                        # 主执行入口
├── Pipfile                        # Python 依赖定义
└── test.py                        # 临时测试脚本
```

## 核心流程

主流程位于 [main.py](main.py)。执行过程大致如下：

1. 读取命令行参数
2. 读取 [config.yaml](config.yaml) 中的默认配置
3. 加载一个或多个 CSV 测试集
4. 并发调用待测 Agent，获取回答摘要
5. 将测试样本组装成 `deepeval` 的 `LLMTestCase`
6. 根据配置选择评测指标执行评测
7. 将结果写入输出 CSV
8. 结果汇总分析输出报告（TODO）

## 主要模块说明

### 1. 入口脚本

- [main.py](main.py)
  - 定义命令行参数 `--csv_path`、`--metrics`
  - 支持单个 CSV 文件评测
  - 未传入 CSV 时，从配置中读取默认测试集路径
  - 并发执行 Agent 调用与测试用例（LLM_test_case）组装
  - 调用 `deepeval.evaluate(...)` 执行评测

### 2. Agent 调用层

- [agents/http_agent.py](agents/http_agent.py)
  - 通过 `POST` 请求调用外部 Agent 接口
  - 请求体格式为：

```json
{
  "query": "用户问题",
  "mode": "DEEP"
}
```

  - 当前从响应中提取 `data.summary` 作为后续评测使用的回答文本

### 3. 评测层

- [evaluator/metrics.py](evaluator/metrics.py)
  - 注册了两个当前实际使用的指标：
    - `reverse_validation`
    - `contextual_recall`
  - `reverse_validation` 基于 `GEval` 自定义，反向校验，用于检查回答是否触犯 `negative_criteria`
  - `contextual_recall` 基于 `deepeval.metrics.ContextualRecallMetric`，正向校验，与预期结果的一致性

- [evaluator/claud_judge_llm.py](evaluator/claud_judge_llm.py)
  - 封装 Judge 模型
  - 当前默认使用 `AnthropicModel`
  - 模型参数来自 [config.yaml](config.yaml)
  - 认证与网关地址来自 `.env`

- [evaluator/deepeval_patch.py](evaluator/deepeval_patch.py)
  - 对 `deepeval` 使用 Anthropics SDK 时的兼容性问题做 monkey patch
  - 包括：
    - 移除不兼容的 `thinking` 参数
    - 跳过 `ThinkingBlock`，提取真实文本
    - 加强 JSON 解析与修复能力

### 4. 工具层

- [tool/config_reader.py](tool/config_reader.py)
  - 单例配置读取器
  - 支持通过 `a.b.c` 形式读取嵌套配置

- [tool/csv_reader.py](tool/csv_reader.py)
  - 读取 CSV 为字典列表

- [tool/csv_writer.py](tool/csv_writer.py)
  - 将评测结果写回 CSV
  - 支持按字段写入和过滤不可序列化字段

## 环境要求

- Python 3.12
- Pipenv

主要依赖见 [Pipfile](Pipfile)：

- `deepeval`
- `anthropic`
- `requests`
- `aiohttp`
- `pyyaml`
- `python-dotenv`
- `json-repair`

## 安装与初始化

### 1. 安装依赖

```bash
pipenv install
```

### 2. 配置环境变量

参考 [ .env.example ](.env.example) 创建 `.env`：

```env
ANTHROPIC_BASE_URL="your-base-url"
ANTHROPIC_AUTH_TOKEN="your-token"
```

### 3. 检查配置文件

编辑 [config.yaml](config.yaml)，重点关注：

- `agents.http_agent.endpoint`: 待测 Agent 接口地址
- `agents.http_agent.mode`: Agent 调用模式
- `judge_llm.anthropic.model`: 评测用 Judge 模型
- `dataset.default_dataset_path`: 默认测试集路径
- `dataset.dataset_metrics_map`: 不同测试集对应的指标

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
  - 可选值：`reverse_validation`、`contextual_recall`
  - 指定了 `--csv_path` 时，该参数必传
  - metrics需要在[evaluator/metrics.py](evaluator/metrics.py)注册并在main中完成map映射

## 配置说明

[config.yaml](config.yaml) 当前包含以下主要配置块：

### Agent 配置

```yaml
agents:
  type: http
  http_agent:
    endpoint: http://...
    mode: DEEP
    call_agent_th_max: 8
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
  run_async: True
  max_concurrent: 10
  throttle_value: 0
```

### 数据集配置

```yaml
dataset:
  default_dataset_path: test/test_cases_robustness.csv
  metrics_if_specify_csv:
    - reverse_validation
    - contextual_recall
```

## 测试集格式

测试集命名规则：test_case_xxx.csv，必须使用**“test_case”**开头且是**csv格式文件**。

测试集 CSV 目前至少包含以下字段，示例可见 [test_suite/test_cases_base.csv](test_suite/test_cases_base.csv)：

- `id`
- `domain`
- `dimension`
- `difficulty`
- `question`
- `expected_answer`
- `source_docs`
- `negative_criteria`

其中：

- `question`：发送给 Agent 的问题
- `expected_answer`：期望回答要点
- `negative_criteria`：不应出现的错误信息，用 `|` 分隔
- `source_docs`：用于说明样本依据来源

## 输出结果

当前代码会将结果写入：

- [test/test_output.csv](test/test_output.csv)

写回字段通常包括：

- 原始测试字段
- `agent_response`
- `is_success`
- 各指标对应的：
  - `*_is_success`
  - `*_score`
  - `*_threshold`
  - `*_reason`

## TODO

- [ ] 测试结果汇总统计生成报告，部分评估指标落地
- [ ] metircs类型完善
- [ ] 文档召回相关（当前好像没办法指定agent输出哪些额外的内容）
- [ ] 多轮对话相关
