## 架构速览

Agent 评测流水线（[main.py](main.py) 的 EvaluationPipeline）：用例准备 →（占位符填充）→
Agent 并发调用 → LLM 评测 → 结果写入与报告。多 Agent 通过 [agents/factory.py](agents/factory.py)
工厂注册（新增 Agent 类需在 `AGENT_CLASS_MAP` 注册 + config 的 `class_config` 配置）。
模块划分与数据流详见 README「项目结构 / 核心流程 / 主要模块说明」。

## Agent skills

### Issue tracker

Issues live as markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root + `docs/adr/`. See `docs/agents/domain.md`.

### 占位符填充

测试数据分两层：raw 模板（含 `[xxx]` 占位符，如 `test_suite/dataqa/dataqa_raw/`）
与填充产物（output_dir，评测数据）。实体来源为 `entity_mapping.json`，可用
`--refresh-entities` 从业务平台采集更新。详细规则（映射表、行内一致性、锚定关联
抽取、fail fast、seed 复现）见 README「占位符填充」章节。

改动占位符映射表或新增占位符类型时，需同步维护 `pipeline/placeholder_filler.py`
的 `ENTITY_TOKEN_MAP`（含 query 取值与 params 参数键映射）与时间占位符规则
（固定白名单 + 泛化写法）；新增数据集占位符前先确认映射覆盖，避免触发 fail fast 中断。
