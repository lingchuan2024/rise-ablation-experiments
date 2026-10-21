# 严格对照的冻结、预检和运行

本驱动只连接已经提供的 `Qwen3.5-35B-A3B` vLLM 服务，不启动服务、不释放他人的 GPU、不下载模型、不持久化凭据。A 为官方结构化 RISE；B 仅增加独立 rerank 的接口及必要说明。全部执行配置由同一个 JSON 对象传给两组，不能按组覆盖。

共同预算为上下文 262144、单次输出 32768（含思考）、每题 100 次模型调用、3600 秒；API 尝试次数为 1、题目外层尝试次数为 1；不强制补终答。使用现有 `official-thinking-v1` 适配和 `level0`，不改变 `rg`、`/tmp`、`read`、截断或自然错误处理。按最新要求，并发上限为 A/B 合计 4。两组同时与官方实现存在的服务器适配须在实验报告中说明。

先在独立目录执行离线检查；这不会发送模型请求：

```bash
python strict_experiment/runner.py offline-preflight
python strict_experiment/runner.py template > reports/strict_qwen35_config.json
```

模板的 `/REQUIRED/` 路径和模型 revision 必须替换为经过实际核实的资源，不能把模板称为已就绪配置。不得把 API key、密码或带凭据 URL 写入配置。服务配置、tokenizer 配置、模型权重、语料及索引均单独绑定哈希。

资源内容收据必须来自真实文件字节。收据保存在被扫描目录之外，避免收据本身改变资源全集：

```bash
python strict_experiment/runner.py write-resource-receipt --kind corpus --root /actual/structured_corpus --output /actual/reports/corpus_receipt.json
python strict_experiment/runner.py write-resource-receipt --kind document_index --root /actual/document_index --output /actual/reports/index_receipt.json
python strict_experiment/runner.py write-resource-receipt --kind model_weights --root /actual/model_snapshot --model-revision ACTUAL_40_CHARACTER_REVISION --output /actual/reports/model_receipt.json
```

收据包括目录全集、每个文件的大小和 SHA256、全集内容哈希。冻结时检查实际文件字节，防止仅填写一张“已验证”收据。语料中的 `.txt` 文件集合必须与 100195 文档的 filename map 完全一致；额外的非 `.txt` 元数据也必须进入完整收据并验证字节，单独记录数量。模型收据必须对应指定模型及同一 revision。重新检查时，只有同进程内文件 inode、大小、mtime、ctime 和收据都未变，才复用已完成的字节校验。`rg_binary` 必须是未经修改的沙箱真正使用的二进制：bubblewrap 的 `code/bin/rg` 优先，其次固定沙箱 PATH 中的系统 `rg`；Docker 则核验固定镜像内的二进制哈希。

先由部署身份工具把实际 PID、启动 ticks、命令行、模型目录、parser、模板、关键 vLLM 源码和包版本的白名单收据写入 `service_config.actual_identity`。随后固定配置中的 `capacity_report` 新路径；服务配置此后不可再改。容量探针不会依赖待生成的容量报告：

```bash
python strict_experiment/capacity_probe.py --prepare-only --config reports/strict_qwen35_config.json
python strict_experiment/capacity_probe.py --execute --config reports/strict_qwen35_config.json --output /actual/reports/strict_capacity_report.json
```

`--output` 必须与配置中的 `capacity_report` 路径一致。C4 探针只用中性合成记录，共两波八次模型请求：先 `163840+65536+65536+65536` 输入 token，再 `229056+163840+163840+163840`，每次最多生成 128 token。每波与同一个真实 BF16 reranker 的 1000 个精确 128-token 段落评分交叠，四个 lease 共用单一模型。新版本为 `strict-qwen35-c4-shared-reranker-capacity-v1`；旧 C3 报告原样保留，不能作为 C4 通过证据。沿用实际准入探针：在屏障后等待实际 `/metrics` 的 `num_requests_running>0`，记录等待开始、真实准入时刻及 running/waiting 计数，随后才开始 rerank；等待上限为 60 秒或更短的请求 timeout。大请求到达 HTTP 层、仍在处理正文或 tokenizer 阶段都不能算引擎已经运行。没有观测到准入时明确失败，不执行评分；即使观测到准入，评分时段仍必须有独立 telemetry 实样本证明 LLM/rerank 交叠。验证实际 tokenize/usage、模型身份、GPU 峰值和服务 running/queue 指标，记录长序列的并发准入、排队或抢占。性能探针的 `length` 停止单独标记，不是研究答题结果。

这只认证两种被测负载，不宣称四个完整 262k 历史可同时驻留，也不宣称实际生成了最坏 32768 token。`229056` 已预留正式输出预算、256 token margin 和 64 token 校准容差。容量报告绑定配置、实际服务配置和全部运行源码哈希，`prepare` 会拒绝缺失、失败或身份不匹配的报告。

容量通过后冻结资源和配置，再做真实服务、沙箱、工具和 reranker 预检：

```bash
python strict_experiment/runner.py prepare --config reports/strict_qwen35_config.json
python strict_experiment/runner.py live-preflight
python strict_experiment/runner.py run-all
```

`run-all` 先运行原始固定 100 题文件中的前 10 题，A/B 各 10，共 20 次。执行检查通过后，自动继续剩余 90 题，A/B 各 100，共 200 次；配置不变，前 20 条终态直接计入正式结果。检查只核验实际发送的系统提示词、工具定义、用户题目、模型身份、配置和 token 账本，不使用正确率决定是否继续。答错、空终答、达到轮次或自然上下文限制都保留，不为改善成绩重跑。

需要分阶段调试驱动时，等价入口为：

```bash
python strict_experiment/runner.py run --stage pilot
python strict_experiment/runner.py run --stage full
python strict_experiment/runner.py status
```

配置和源码冻结在 `reports/strict_qwen35/run_manifest.json`，A/B 准备契约在相邻 `contracts/`。真正发送请求时保存完整提示词、工具和用户题目到 `runs/strict_qwen35/{A,B}/request_contracts/`，每次模型调用的身份、参数及用量在 `api_ledgers/`。题目终态位于 `_per_query/`，通过原子仅创建写入，禁止覆盖；评分结果另存，绑定终答 SHA，不改写终态。私有轨迹和断点保留在运行服务器，不进入 GitHub。

每题终态已安全写入后，两组都执行相同的收尾维护：先把原文硬链接的路径、原始语料路径、冻结源 SHA 和派生文件清单写入不可覆盖的 `_maintenance/source_manifests/`，再释放该题 `_working/` 和 `_rerank_runtime/`，避免 200 题累计目录和派生缓存耗尽 inode。原始语料、终态、轨迹、断点、请求和 token 账本全部保留。该维护不发生在题目运行中；被中断的题目不清理；收尾中断可根据已保存清单继续清理，不能重跑终答。需要复查目录时可从冻结语料重新建立原文硬链接，派生缓存由断点及固定段落索引重建。

基础设施中断需要先排查原因，随后使用 `run-all --resume-infrastructure` 或相应阶段的同名参数。已有自然终态始终直接复用；已有调用痕迹但断点丢失的题目禁止从头重跑。恢复时核验模型、题目、提示词、工具和运行配置身份，并沿用断点中的已消耗预算。资源、配置、提示词或源码变化均要求新建实验，不得续接原终态。

驱动测试：

```bash
python -m unittest strict_experiment.test_runner -v
```

测试覆盖自然错误终态不可覆盖、首 10 题保留且全量仅新增 180 次、实际请求提示词/模型漂移、token 账本漂移、断点缺失拒绝重跑、伪资源收据以及文件字节或目录范围变化。离线测试通过不代表服务或实验已运行。
