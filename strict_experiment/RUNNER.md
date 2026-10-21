# 严格对照的冻结、预检和运行

本驱动只连接已经提供的 `Qwen3.5-35B-A3B` vLLM 服务，不启动服务、不释放他人的 GPU、不下载模型、不持久化凭据。A 为官方结构化 RISE；B 仅增加独立 rerank 的接口及必要说明。全部执行配置由同一个 JSON 对象传给两组，不能按组覆盖。

共同预算为上下文 262144、单次输出 32768（含思考）、每题 100 次模型调用、3600 秒；API 尝试次数为 1、题目外层尝试次数为 1；不强制补终答。使用现有 `official-thinking-v1` 适配和 `level0`，不改变 `rg`、`/tmp`、`read`、截断或自然错误处理。并发上限为 A/B 合计 3。两组同时与官方实现存在的服务器适配须在实验报告中说明。

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

收据包括目录全集、每个文件的大小和 SHA256、全集内容哈希。冻结时检查实际文件字节，防止仅填写一张“已验证”收据。语料文件集合必须与 100195 文档的 filename map 完全一致，模型收据必须对应指定模型及同一 revision。重新检查时，只有同进程内文件 inode、大小、mtime、ctime 和收据都未变，才复用已完成的字节校验。`rg_binary` 必须是未经修改的沙箱真正使用的二进制：bubblewrap 的 `code/bin/rg` 优先，其次固定沙箱 PATH 中的系统 `rg`；Docker 则核验固定镜像内的二进制哈希。

资源和配置核实后冻结，再做真实服务、沙箱、工具和 reranker 预检：

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

基础设施中断需要先排查原因，随后使用 `run-all --resume-infrastructure` 或相应阶段的同名参数。已有自然终态始终直接复用；已有调用痕迹但断点丢失的题目禁止从头重跑。恢复时核验模型、题目、提示词、工具和运行配置身份，并沿用断点中的已消耗预算。资源、配置、提示词或源码变化均要求新建实验，不得续接原终态。

驱动测试：

```bash
python -m unittest strict_experiment.test_runner -v
```

测试覆盖自然错误终态不可覆盖、首 10 题保留且全量仅新增 180 次、实际请求提示词/模型漂移、token 账本漂移、断点缺失拒绝重跑、伪资源收据以及文件字节或目录范围变化。离线测试通过不代表服务或实验已运行。
