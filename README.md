# rga_agent_eval

一个基于 `deepeval` 的 Agent 评测脚本项目,支持异步并发评测和结果统计分析。

## 项目结构

```text
.
├── agents/                        # Agent 注册目录
│   ├── __init__.py                # 包初始化
│   ├── factory.py                 # Agent 工厂（根据配置创建实例）
│   └── http_agent.py              # HTTP 方式调用待测 Agent
├── evaluator/                     # LLM-as-Judge 评测层
│   ├── claud_judge_llm.py         # Claude Judge 模型封装
│   ├── deepeval_patch.py          # deepeval/anthropic 兼容性 patch
│   ├── metrics.py                 # 自定义评测指标注册与构造
│   └── runner.py                  # 逐例异步评价、统一限流、重试与串行复核
├── pipeline/                      # 流水线模块
│   ├── __init__.py                # 包初始化
│   ├── test_case_loader.py        # 测试用例加载器（CSV 解析 → 用例列表）
│   ├── resume.py                  # 恢复预检、历史筛选、多轮 tmp 还原
│   └── placeholder_filler.py      # 占位符填充器（[xxx] → 实体/时间值）
├── result/                        # 评测结果输出目录
├── test_suite/                    # 测试数据集存放目录
│   ├── dataqa/
│   │   ├── dataqa_raw/            # DataQA 待填充模板（含 [xxx] 占位符）
│   │   ├── test_cases_*.csv       # 填充后可执行数据集（评测数据）
│   │   └── entity_mapping.json    # 实体映射（业务平台采集 / 手工维护）
│   ├── pvassistant/               # PVAssistant 测试集
│   └── diagnosis/                 # Diagnosis 测试集
├── tool/                          # 工具模块
│   ├── __init__.py                # 工具模块导出
│   ├── async_result_writer.py     # 异步结果写入器
│   ├── business_platform_client.py # 业务平台 API 客户端（采集实体映射）
│   ├── business_platform_token_manager.py # 业务平台登录 token 管理
│   ├── collection_result.py       # 结果统计分析工具
│   ├── concurrency.py             # 通用线程池调度
│   ├── config_reader.py           # 配置读取工具（单例）
│   ├── csv_reader.py              # CSV 读取工具
│   ├── csv_writer.py              # CSV 写入工具
│   ├── result_checkpoint.py       # 批次提交、checkpoint 校验与只读恢复
│   ├── file_utils.py              # 通用文件系统工具
│   ├── markdown_writer.py         # Markdown 报告输出（含 seed 记录）
│   └── get_bad_cases.py           # 失败用例提取工具
├── docs/                          # 项目文档
├── .env.example                   # 环境变量示例
├── config.yaml.example            # 配置文件示例
├── config.yaml                    # 项目配置
├── main.py                        # 主入口（CLI + EvaluationPipeline）
├── Pipfile                        # Python 依赖定义
├── requirements.txt               # pip 依赖列表
└── README.md                      # 项目文档
```

## 核心流程

主流程位于 [main.py](main.py),执行过程如下:

1. **读取配置**: 解析命令行参数和 [config.yaml](config.yaml) 配置
2. **（可选）刷新实体映射**: 配置 `refresh_entities: true` 或传 `--refresh-entities` 时,
   登录业务平台采集实体数据,更新 `entity_mapping.json`
3. **占位符填充**: 嗅探模式下先扫描 raw 模板（`target_dirs` 命中）填充并输出到
   `output_dir`,再加载可执行数据集;`-cp` 指定模板文件时在加载后直接填充
4. **加载测试集**: 从指定路径或默认目录加载 CSV 测试集
5. **并发调用 Agent**: 使用线程池并发调用待测 Agent,获取回答
6. **组装测试用例**: 将测试样本组装成 `deepeval` 的 `LLMTestCase`
7. **执行评测**: 根据配置的指标执行异步并发评测
8. **批次提交结果**: 持续评价并接收终态用例，累计约 20 条交给写入线程保存 CSV 和 checkpoint
9. **统计分析**: 所有测试完成后,统计各指标的成功率并输出报告（含占位符填充 seed）

## 主要模块说明

### 1. 入口脚本 ([main.py](main.py))

- 定义命令行参数 `--csv_path`、`--metrics`、`-a`、`--seed`、`--fill-preview`、`--refresh-entities`、`--resume`、`--resume-result-dir`
- 支持单个 CSV 文件评测、目录嗅探批量评测、断点重入（`--resume`）
- 多 Agent 架构：按配置的 enabled 类（PVAssistant / Diagnosis / DataQA）分组执行
- 使用线程池并发执行 Agent 调用
- 直接调用 DeepEval 指标的 `a_measure()`，逐例完成评价和复核，不等待全量主评
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

### 全流程自动评测（嗅探模式）

```bash
pipenv run python main.py
```

不指定 `-cp` 时按 `dataset.default_dataset_path` 嗅探各 enabled 类子目录的
`test_cases_*.csv`。启用占位符填充时,先扫描 raw 模板（`target_dirs` 命中）填充并
输出到 `output_dir`,再评测生成的可执行数据集。

### 指定测试集与指标执行

```bash
pipenv run python main.py \
  --csv_path test_suite/dataqa/dataqa_raw/test_cases_direct_inquiry.csv \
  --metrics dataqa_capability dataqa_params
```

### 填充预览（只替换不评测）

```bash
pipenv run python main.py --fill-preview \
  --csv_path test_suite/dataqa/dataqa_raw/test_cases_direct_inquiry.csv
```

### 参数说明

- `--csv_path` / `-cp`
  - 指定单个 CSV 测试文件
  - 不传时使用配置中的默认路径（嗅探模式）

- `--metrics` / `-m`
  - 正常评测指定 `--csv_path` 时必传（agent-only 除外）
  - resume 优先使用 `-m`，否则读取 tmp meta 中的 metrics，最后使用配置兜底；仍缺失则报错
  - 新增 metric 需要在 [evaluator/metrics.py](evaluator/metrics.py) 注册并在 main.py 中完成映射

- `-a` / `--agent_classes`
  - 指定调用的 agent 类名,支持多个（如 `-a Diagnosis`、`-a PVAssistant DataQA`）
  - 正常评测不传时使用配置中 `enabled: true` 的类
  - resume 不传时扫描标准目录下有 tmp 的已注册 Agent，不受 enabled 限制；指定时忽略名称大小写

- `--seed N`
  - 占位符填充随机种子;不传则随机生成并输出到日志和评测报告
  - 同一 seed + 同一模板重跑可复现相同测试数据

- `--fill-preview`
  - 仅执行占位符填充,产物输出到 `placeholder_fill.output_dir`,不执行评测

- `--refresh-entities`
  - 填充前先登录业务平台更新 `entity_mapping.json`（登录/接口失败时中断）
  - 与配置项 `placeholder_fill.refresh_entities` 取 OR

- `--resume`
  - 从 tmp 中间文件恢复评测,跳过 Agent 调用阶段
  - 不重新填充,seed 从 tmp meta.json 读取并输出到报告
  - 不指定历史目录时，全量重新评价所选范围内可评价用例，不读取历史结果
  - 不能与 `--fill-preview` 同时使用

- `--resume-result-dir`
  - 可选，仅与 `--resume` 配合：指定一个 `result/<时间戳>` 目录，校验 checkpoint 并复用通过结果
  - 不自动寻找最新目录；路径不存在、非目录或为空时立即报错

### 恢复评价：全量重评与增量恢复

```powershell
# 从 tmp 全量重新评价：不读取任何历史 CSV 或 checkpoint
python .\main.py --resume -a Diagnosis

# 增量恢复：将下面的时间戳替换为实际已存在的结果目录
python .\main.py --resume -a Diagnosis --resume-result-dir ".\result\<时间戳>"

# 只恢复一个标准 Agent 目录或其中一份 tmp 文件
python .\main.py --resume -cp ".\test_suite\diagnosis" --resume-result-dir ".\result\<时间戳>"
python .\main.py --resume -cp ".\test_suite\diagnosis\tmp\test_cases_pekat_en_tmp.csv" --resume-result-dir ".\result\<时间戳>"
```

只支持以下标准目录（`test_suite` 对应 `dataset.default_dataset_path`）：

```text
test_suite/<agent>/tmp/test_cases_<数据集>_tmp.csv
test_suite/<agent>/tmp/test_cases_<数据集>_tmp.meta.json
result/<时间戳>/<agent>/result_outputs_<数据集>.csv
result/<时间戳>/<agent>/result_outputs_<数据集>.checkpoint.json
```

不递归扫描人工临时目录、备份或旧式平铺结果。`-cp` 保留实际 Agent 归属，不再归入
`default`；同时传 `-a` 时必须且只能指定同一个 Agent。历史结果只在所选 Agent 的
同名目录内寻找，不跨 Agent 回退。未选择的 Agent、其他数据集不会因缺 checkpoint 阻断本次恢复。

数据集以 tmp 的标准文件名及 meta/`_source_csv` 对应，原始测试集即使已移动也不要求
仍然存在；来源文件名冲突则报错。用例优先按 `test_id`、`用例编号`、`case_id`、`id`
（多轮也支持父用例编号）对应；缺少编号时按 query 对应，不新增 query 重复检查。
匹配后还会比对语言及实际评价输入，不复用错配或已经改变输入的旧结论。

增量恢复在组装用例和调用模型前，先完成所选数据集的历史预检：

| 历史状态 | 行为 |
| --- | --- |
| CSV、checkpoint 都不存在 | 视为尚无历史结果，从 tmp 正常评价 |
| CSV 存在、checkpoint 缺失 | 旧格式不兼容，fast fail；可不传历史目录进行全量重新评价 |
| 有合法的 0 条提交 checkpoint | 尚无可信结果，包括首批写入中断的情况，重新评价 |
| 非零 checkpoint，但 CSV 缺失、记录不足或摘要不符 | fast fail，不猜测修复 |
| checkpoint 已提交范围之后有尾部内容 | 不采纳尾部，对应 tmp 用例重新评价，不修改历史文件 |
| 匹配且可信的历史 `is_success=True` | 复用，不进入用例组装和评价 |
| 历史未通过、整体状态为空或无匹配记录 | 使用 tmp 输入重新评价 |

是否通过只看最终 `is_success`，不根据 `evaluate_error`、单项指标或复核模型状态另行
推翻整体结论。评价口径与其他约定条件的一致性由用户保证，不进行配置指纹校验。

多轮 tmp 中的 `(tN)` 字段会反向还原为各轮用例；整体未通过时重评整个父用例，
不只重评失败轮。任一轮所需响应为空/缺失时，整个父用例跳过评价模型，保留原始内容、
缺失轮次错误和整体 `is_success=False`；其余轮次不伪造评价结论。单轮缺失响应同样跳过。
DataQA 使用其结构化响应字段，检索轨使用保存的检索结果，不重新请求 Agent 或检索服务。

如果所选 tmp 用例全部有可信的通过结果，仅提示“全部通过，无需恢复评价”并退出，
不创建新结果目录、CSV、checkpoint 或报告。否则，新结果包含“复用通过记录 + 本次重评结果 + 不可评价失败记录”，
统计和报告覆盖本次完整输入范围。全部不可评价不等于全部通过，
仍会保存失败结果。复用记录不消耗模型请求，后续追加不会覆盖它们或重复保留历史失败行。

新运行使用独立目录，同秒重名自动增加后缀；历史结果与输入 tmp 保持不变。

### 仅调用 Agent（agent-only）

在目标 Agent 的 `class_config` 下开启 `is_agent_only`：

```yaml
agents:
  http_agent:
    class_config:
      Diagnosis:
        enabled: true
        is_agent_only: true
```

然后按正常入口执行：

```bash
pipenv run python main.py -a Diagnosis
```

该模式会完成 Agent 调用并将响应写入测试数据集同级的 `tmp/` 目录，随后跳过
LLMTestCase 组装、评测、结果目录与报告生成。`--resume` 不受该配置影响，仍用于从
已有 tmp 中间文件恢复评测。agent-only 不依赖评测指标：指定单个 CSV 时可以省略
`-m/--metrics`，目录嗅探时也不要求该数据集存在 `dataset_metrics_map` 映射。

```bash
pipenv run python main.py -a Diagnosis -cp path/to/test_cases_diagnosis.csv
```

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
  run_async: True          # False 时同时执行的指标最多 1 个
  max_concurrent: 10       # 主评/检索/结构化/复核/重试共用的指标执行并发上限
  throttle_value: 0        # 所有指标尝试共用的启动间隔（秒）
  metric_timeout: 180      # 单次 a_measure 超时（秒），不含排队及重试退避
```

**重要**: `throttle_value: 0` 可能导致触发 API 速率限制,建议设置为 1-2 秒。

同一用例的各指标依次执行；全部主评指标完成后判断是否复核。整体主评通过则直接产出终态；
不通过且已配置复核模型时，依次执行该用例的 model2、model3，然后按原有规则汇总：
每个复核指标至少两个模型明确返回布尔 True 才通过。其他用例无需等待它的主评、复核或重试。
多轮父用例内各轮依次处理，所有轮次结束后才产出一条终态结果。

最多保留 `2 × 有效并发数` 个在途父用例；每次指标尝试使用独立指标实例，模型类型、提示词、
阈值和 GEval 原生评分路径保持不变。缺失输入/最终调用失败的指标记为失败，不能被其他指标的通过掩盖。
缺失有效 score、异常及超时按 `retry.eval_max_retries` 重试；项目级退避使用异步等待并释放指标配额。
配额覆盖整个 `a_measure`，不是严格的 HTTP 请求速率限制；指标内部的模型请求及 SDK 自带重试
仍处于该配额内。`run_async: False` 仍使用异步接口，只把有效并发限制为 1。

主链路不再调用 `deepeval.evaluate()`，因此不产生其测试运行缓存、Rich 逐条进度任务或
Confident AI 的运行级上传；使用项目 CSV、checkpoint 和报告作为结果依据。
原 `deepeval_patch` 的进度兜底保留，供独立调试脚本仍使用 `evaluate()` 时兼容。

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

### 占位符填充配置

```yaml
dataset:
  placeholder_fill:
    enabled: true                          # 是否执行占位符替换
    refresh_entities: false                # 是否在替换前刷新实体映射表（登录/接口失败中断）
    entity_mapping_path: test_suite/dataqa/entity_mapping.json
    target_dirs:                           # raw 模板目录路径片段（命中即参与填充）
      - dataqa_raw
    output_dir: test_suite/dataqa          # 填充产物输出目录（原文件名，同名覆盖）
```

### 业务平台配置（实体采集）

```yaml
business_platform:
  login_url: "https://pmms01-test.rundoai.com/api/system/login"
  username: "your_username"
  password: "your_password"
  api_base_url: "https://pmms01-test.rundoai.com"
  device_page_size: 10   # 每类设备采集条数
```

### 结果保存配置

```yaml
result:
  save_path: result/  # 结果输出目录
  write_batch_size: 20  # 每批落盘的用例数，整数 10–30，未配置默认 20
```

评价持续补充新用例，不再按写入批次分段评价。完成各指标、失败重试和所需复核的终态结果，
按**完成顺序**累计到 `write_batch_size` 条后提交写入队列；慢用例不阻塞其他终态结果落盘。
独立写入线程串行提交 CSV 和 checkpoint；评价过程不逐批等待 `flush()`，数据集结束时才等待写入收尾。
写入队列最多等待 2 批，队列满时异步背压暂停补充新用例，但不阻塞事件循环；已有在途任务可以完成。
除原有输入/最终汇总数据外，等待提交的结果最多为 1 个正写批次、2 个排队批次、1 个缓冲批次，
外加最多 `2 × 有效并发数` 个在途父用例。此上限按条数计算，不是字节数上限。
正常结束或一次 Ctrl+C 软中断时，已完成的不足一批结果也会写入；未完成主评/复核的父用例不会伪造终态。
多轮用例按父用例计为 1 条，整组处理后合并输出，不拆散到不同批次。
各轮通过状态和错误信息保持独立，整体状态单独记录为 `_parent_all_pass`；
合并结果的 `is_success` / `用例是否通过` 取整体状态，任一轮未通过则整条用例未通过。
终端每秒显示终态数量、待启动数量、主评/复核执行数、重试等待数，以及当前 Agent 累计已提交条数、
待写批次数；每批另有 checkpoint 提交事件。结果 CSV 命名保持不变，行顺序不再保证与输入一致。

后续批次新增的错误、重试或复核字段会自动补入表头，旧行对应列留空；
仅表头扩展时重写已有内容，普通批次直接追加。写入失败会被健康检查发现，停止新评价/复核/重试，
取消仍在等待的任务并报错；已经发送给模型的请求可能仍产生消耗，不能保证故障时仅损失一批结果。
正常评价、全量重新评价和增量恢复的结果写入，均自动维护同名 `.checkpoint.json`。
仅 Agent 响应的 tmp 和报告派生的 `bad_cases_*.csv` 不属于评价提交文件，不生成该检查点。

checkpoint 包含 `version`、`result_file`、`agent`、`last_committed_batch`、`committed_rows`、
`last_case_id`、`committed_columns`、`content_sha256`。其中记录数不含表头，多轮合并后算一条；
最后一个编号只辅助排查，不能替代内容校验，也不是原始输入的续跑游标。
`committed_rows` 表示当前结果文件按完成顺序写入的可信前缀，resume 仍按用例身份和输入内容匹配，
不按输入行号跳过。批次完整不代表其中每条用例都通过。

写入顺序为 **CSV 写入并同步 → checkpoint 临时文件写入并同步 → 原子替换 checkpoint**。
首次写入前先建立 0 条提交检查点。CSV 已写入但 checkpoint 未提交成功时，该批不能复用，
同时停止后续评价。恢复只读取已确认的完整范围，不截断历史文件。

摘要按已提交字段的字符串值和逻辑 CSV 记录计算，支持带换行的大字段，不按物理行数或
单纯字节偏移判断完整性。普通追加增量计算摘要；表头扩展时重建摘要。若扩展表头后提交中断，
仍可按旧 checkpoint 的字段集合验证旧记录，新字段与未提交尾部不被采纳。

## 占位符填充（Placeholder Fill）

评测执行前,将测试数据集中的 `[xxx]` 占位符替换为真实值,使同一份模板数据集可反复
生成不同的可执行数据集。实现位于 [pipeline/placeholder_filler.py](pipeline/placeholder_filler.py)。

### 目录结构

```text
test_suite/
└── dataqa/
    ├── dataqa_raw/          ← 待填充模板（含占位符，不可直接执行）
    │   └── test_cases_direct_inquiry.csv
    ├── test_cases_*.csv     ← 填充后可执行数据集（output_dir 产物，评测数据）
    └── entity_mapping.json  ← 实体映射（业务平台采集 / 手工维护）
```

### 占位符类型

| 类型 | 示例 | 处理方式 |
|---|---|---|
| 实体类 | `` `[电站]` ``、`` `[逆变器设备SN]` `` | 从 entity_mapping.json 按映射表随机抽取 |
| 时间类 | `` `[今天起始]` ``、`` `[本月起始]` ``、`` `[3天前起始]` `` | 基于执行当天动态计算（`%Y-%m-%d %H:%M:%S`） |
| 监控项类 | `[直流电压1,当日发电量]`、`[全部监控项]` | 普通文本（非占位符），裸写原样保留（detailList 参数值） |

占位符标识为反引号包裹的 `` `[电站]` `` 形式，替换时反引号一并去除；
裸 `[电站]` 是普通文本，不参与替换（如 `detailList=[直流电压1,当日发电量]` 参数值裸写即可）。

### 时间占位符

时间占位符分为固定写法和泛化写法两类，基于执行当天动态计算，统一输出格式
`%Y-%m-%d %H:%M:%S`。整个填充过程只捕获一次执行时刻作为基准，全数据集
（query 与 expected_parameters）共用。

下表 token 省略反引号标识，实际书写需以 `` `[xxx]` `` 反引号包裹形式才被识别。

| 占位符 | 计算结果 | 说明 |
|---|---|---|
| `[今天起始]` | 今天 00:00:00 | |
| `[今天结束]` | 今天 23:59:59 | 明天 00:00:00 减 1 秒 |
| `[本月起始]` | 本月 1 号 00:00:00 | |
| `[本月结束]` | 本月最后一天 23:59:59 | 下月 1 号减 1 秒 |
| `[今年起始]` | 今年 1 月 1 日 00:00:00 | |
| `[今年结束]` | 今年 12 月 31 日 23:59:59 | |
| `[上月起始]` | 上月 1 号 00:00:00 | 本月 1 号减 1 天再取 1 号 |
| `[上月结束]` | 上月最后一天 23:59:59 | 本月 1 号减 1 秒 |
| `[去年起始]` | 去年 1 月 1 日 00:00:00 | |
| `[去年结束]` | 去年 12 月 31 日 23:59:59 | |
| `[3天前起始]` | 今天 − 3 天 00:00:00 | 天边界，非执行时刻 −72h |
| `[7天前起始]` / `[一周前起始]` | 今天 − 7 天 00:00:00 | 两个写法等价 |
| `[24小时前]` | 执行时刻 −24h | 唯一滚动窗口 token，保留时分秒 |
| `[当前时刻]` | 执行时刻 | 保留时分秒 |

泛化写法（N 为数字，可含 0；单位必须是"个月"，`[3月前起始]` 不识别）：

| 写法 | 计算结果 | 说明 |
|---|---|---|
| `[N天前起始]` | 今天 − N 天 00:00:00 | 天边界，非执行时刻 −N×24h |
| `[N天前结束]` | 今天 − N 天 23:59:59 | |
| `[N个月前起始]` | 今天 − N 个月（同日）00:00:00 | 日历语义；日号超过目标月最后一天时钳制（如 3-31 减 1 个月 → 2-28/29） |
| `[N个月前结束]` | 同上 23:59:59 | |
| `[N年前起始]` | 今天 − N 年（同月同日）00:00:00 | 闰日钳制（2-29 减 1 年 → 2-28） |
| `[N年前结束]` | 同上 23:59:59 | |

`[3天前起始]` / `[7天前起始]` 固定写法与 `[N天前起始]` 泛化写法结果一致。

匹配规则：

- **标识为反引号包裹的 `` `[xxx]` ``**：只有反引号包裹的 token 才被识别替换，
  替换时反引号一并去除；裸 `[xxx]` 是普通文本，不识别、不报错、原样保留
- **固定写法精确全等匹配**：token 必须与白名单完全一致（如 `[3天前]` 不在白名单内）
- **泛化写法模式匹配**：`[N天前起始/结束]`、`[N个月前起始/结束]`、`[N年前起始/结束]`
  支持任意数字 N（如 `[5天前起始]`、`[2个月前结束]`）；带"起始"/"结束"之外后缀的
  写法（如 `[2周前起始]`、`[3月前起始]`、不带边界的 `[N天前]`）不识别
- **未识别 fail fast**：反引号包裹但内容不识别（非时间、非实体）
  即中断，携带文件、行号、列名报错；裸 `[xxx]` 不触发 fail fast
- **"起始"/"结束"语义**：起始类取边界 00:00:00，结束类取边界 23:59:59；
  仅 `[24小时前]` / `[当前时刻]` 为精确到秒的滚动时刻
- **与参数键无关**：时间 token 的值不随所在参数键变化（实体 token 才按参数键
  取不同字段），query 与 expected_parameters 中同一 token 填出同一值
- **seed 复现例外**：`[当前时刻]` / `[24小时前]` 按执行时刻计算，同 seed 重跑
  仍存在秒级差异；其余 token（含泛化写法）只依赖日期，同一天内可复现

### 实体映射表

query 取人可读字段,`expected_parameters` 按参数键取接口期望字段:

| 占位符 | 数据源分类 | query 取值 | params 取值（按参数键） |
|---|---|---|---|
| `[电站]` | 电站 | `name` | `siteId/powerStationId/ids/siteIds/stationId` → `id` |
| `[项目]` | 项目 | `name` | `settlementOrganizationId` → `id` |
| `[大区/省]` | 电站 | `provinceCityDistrict` 第1段 | 同左（`area`） |
| `[大区/省-州/市]` | 电站 | 前2段 | 同左 |
| `[大区/省-州/市-区]` | 电站 | 全量 | 同左 |
| `[地址]` | 电站 | `provinceCityDistrict` 全量（近似） | 同左（`address`） |
| `[逆变器设备SN]` | Inverter | `deviceSn` | `deviceSn/externalId_like` → `deviceSn`；`deviceId` → `id` |
| `[逆变器设备ID]` | Inverter | `id` | `deviceId` → `id` |
| `[逆变器设备名称]` | Inverter | `name` | `deviceName_like` → `name` |
| `[气象站设备SN]` | MeteorologicalStation | `deviceSn` | `deviceSn`；`deviceId` → `id` |
| `[气象站设备ID]` | MeteorologicalStation | `id` | `deviceId` → `id` |
| `[执行人]` | 消缺工单/消缺记录/巡检工单 | `chargePerson` → `chargePersonName` | `chargePersonId` → `chargePersonId` |
| `[整改人]` | 隐患工单 | `rectifyPersonName` | `rectifyPersonId` → `rectifyPersonId` |
| `[修改人]` | 低效告警白名单 | `changedBy` | `changedUserId/changedBy` → `changedBy` |
| `[消缺工单编号]` | 消缺工单 | `id` | `number` → `id` |
| `[巡检工单编号]` | 巡检工单 | `id` | `number` → `id` |
| `[消缺记录工单编号]` | 消缺记录 | `id` | `number` → `id` |

### 随机与一致性规则

- **每行独立随机**: 不同用例行抽取不同实体（同 seed 可复现）
- **行内一致性**: 同一行内相同 token 用同一记录;同一分类复用同一记录
  （如 `[逆变器设备SN]`/`[逆变器设备ID]`/`[逆变器设备名称]` 指向同一台设备,
  `[执行人]` 与 `[消缺工单编号]` 来自同一条工单）
- **锚定+关联抽取**: 行内出现 `[电站]` 时作为锚实体,其余实体优先从锚的关联记录
  中抽取（外键: `powerStationIds` / `siteId` / `stationId` / `powerStationId` /
  `deviceId`）;锚下无关联记录时降级全局随机并输出 warning
- **seed 可复现**: 同一 seed 重跑实体抽取完全一致（`[当前时刻]`/`[24小时前]` 按
  执行时刻计算,存在秒级差异）;seed 输出到评测报告标题区,`--resume` 时从
  tmp meta.json 读取保持一致

### 异常处理

| 场景 | 行为 |
|---|---|
| 反引号包裹但未识别的占位符（非实体、非时间） | **fail fast 中断**,报错含 token、文件、行号、用例编号、列名 |
| 实体映射文件缺失/格式错误 | 中断,提示检查 `entity_mapping_path` |
| `--refresh-entities` 登录或接口失败 | **中断执行**（不静默回退旧映射） |
| 锚定关联无匹配记录 | 降级全局随机 + warning（不中断） |
| 已填充产物被指定为填充输入（-cp output_dir 文件） | 拒绝并提示指定 raw 模板 |
| 嗅探 output_dir 内的产物 | 自动跳过（不二次填充、不重复评测） |

### 实体映射刷新

`--refresh-entities` 或 `refresh_entities: true` 时,填充前调用
`BusinessPlatformClient().export_entity_mapping()` 更新实体映射:

- 采集内容: 电站/项目、设备（Inverter/气象站）、消缺工单、消缺记录、巡检工单、
  隐患工单、低效告警白名单
- 已存在的手工分类（区域/地址等）保留,仅更新采集到的分类
- 也可独立执行: `python -m tool.business_platform_client --export-mapping`

## 测试集格式

测试集命名规则:`test_cases_*.csv`,必须以 `test_cases_` 开头且为 CSV 格式。

Diagnosis 数据集可通过 `language` 列指定响应语言：`zh-CN` 或 `en-US`。调用时该值会写入
请求头 `Language`；列缺失或值为空时默认使用 `zh-CN`。
Diagnosis 默认使用 `response_mode=streaming`，客户端从 SSE 的
`complete.data.content` 提取最终文本并写入用例的 `agent_response` 字段；同时保留普通
JSON 响应兼容。

诊断首轮响应含 `<suggest>` 时，客户端从原始 query 的 `model:`（或“型号是／型号：”）
提取型号，与候选中的 `Model (...)`／`型号（...）` 完整值比较。仅去除首尾空白，
区分大小写并保留连字符和后缀；`SG40CX` 与 `SG40CX-P2` 不会相互匹配。
唯一命中时，以该标签内的完整文本（保留编号、去除标签及首尾空白）再次请求，
复用首轮 `X-Session-Id`、语言、用户和响应模式。每次调用最多两轮；最终一轮正文
写入 `agent_response`，`res_time(s)` 统计整个调用耗时。
型号缺失、解析失败、零匹配或多匹配时保留首轮响应并记录原因；第二轮仍含候选时
直接返回，不继续选择。第二轮请求异常沿用外层重试机制，默认重新开始整个流程。

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
