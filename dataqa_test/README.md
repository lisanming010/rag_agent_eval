# 临时问数智能体测试

该目录只保留临时测试的入口和结果整理逻辑。HTTP 调用、鉴权、配置、并发、
日志均复用项目公共模块，不在这里维护另一份 DataQA 实现。测试用例统一使用
JSON/JSONL 格式，存放在当前目录的 `cases/` 子目录中。

## 运行

复制独立鉴权配置并填写密码：

```bash
cp dataqa_test/auth.yaml.example dataqa_test/auth.yaml
```

`auth.yaml` 已被忽略。也可使用 `DATAQA_AUTH_PASSWORD`、
`DATAQA_AUTH_USERNAME`、`DATAQA_AUTH_LOGIN_URL`、`DATAQA_AUTH_TENANT_ID`、
`DATAQA_AUTH_ACCEPT_LANGUAGE` 和 `DATAQA_AGENT_ENDPOINT` 环境变量覆盖。

临时测试使用独立的 `RefreshingDataQA` 子类，不修改项目原 DataQA。首次请求前登录并
提取 `data.access_token`；问数请求同时设置 `Authorization: Bearer <JWT>`、
`X-Business-Token: <JWT>`、固定租户 ID 和 `Accept-Language: en-US`。收到 401 时，
线程安全地刷新 token 并重试一次；并发请求会复用其他线程已经刷新的 token。

从项目根目录执行：

```bash
python dataqa_test/main.py
```

默认只执行 `required_data_domains` 包含 `mysql` 的用例。传入 `all` 可关闭筛选：

```bash
python dataqa_test/main.py --data-domain all
```

默认读取 `dataqa_test/cases/` 下的全部 `.json` 和 `.jsonl` 文件。也可指定单个
用例文件、问题字段和输出目录：

```bash
python dataqa_test/main.py \
  --input dataqa_test/cases/plant-overview.jsonl \
  --output-dir dataqa_test/output \
  --data-domain mysql \
  --query-field 示例提问 \
  --max-workers 3 \
  --submit-delay 1
```

JSON 文件支持顶层数组：

```json
[
  {"id": 1, "query": "查询电站总览"}
]
```

或包含 `cases` 数组的对象：

```json
{
  "cases": [
    {"id": 1, "query": "查询电站总览"}
  ]
}
```

JSONL 文件每行是一条用例。当前用例的 `turns` 为问题数组；单轮和多轮都会按
数组顺序执行，同一条用例的多轮请求复用 `session_id`：

```json
{"id":"PO-004","turns":["Show the current Plant Overview."]}
```

每次运行会创建独立目录：

```text
dataqa_test/output/<run_id>/
├── manifest.json
├── results.jsonl
└── raw/
    ├── PO-004.json
    └── CR-101.json
```

- `manifest.json`：运行时间、实际并发配置、输入数量和状态统计。
- `results.jsonl`：每条用例一行，按 `case / execution / turns` 分层保存精简结果。
- `raw/`：完整接口响应。每个用例 ID 对应一个 JSON 文件，文件内按轮次和请求顺序
  保存 `requests[]`。每条请求记录阶段（`new_query` / `confirm_execute`）、外层重试
  序号、鉴权重试序号、HTTP 状态码、原始响应和异常；所有实际发出的问数请求都会记录。
  登录响应和鉴权请求头不会写入 raw，避免泄露 JWT。

请求线程完成一条用例后会立即将结果放入有界队列，由单独的写入线程生成 raw 并
追加、刷新 `results.jsonl`。写入过程与后续请求并行执行；队列满时自动施加背压，
避免 raw 响应在内存中无限累计。每次运行开始时会原子创建 `run_id` 目录，避免两个
进程误用相同目录。

每轮 `answer` 保存 Markdown 文本、组件类型、`has_echart` 和精简表格摘要；
`echart.option`、完整表格行及其他大字段仅保留在 raw。主结果的
`execution.raw_ref` 指向对应的用例 raw 文件。状态区分 `success`、
`business_warning`、`request_failed`、`new_query_failed`、`partial_success` 和
`skipped`。只要有一条用例未成功，入口退出码为 1。

## Resume

当前临时测试入口不支持 `--resume`。执行中断时，已经由写入线程完成的 raw 和
`results.jsonl` 行会保留，但再次启动仍会创建新运行并执行全部输入用例。项目主评测
入口的 `--resume` 不能用于该临时测试目录。
