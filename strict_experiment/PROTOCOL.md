# 官方结构化 RISE：仅增加独立 rerank

目标：测量 **Qwen3.5-35B-A3B 在官方结构化 RISE 配置下增加独立 rerank 工具及必要说明的增量收益**。工具说明本身是可用工具接口的一部分；这轮不分离神经模型、候选选择、工具说明各自的贡献。

| 条件 | A | B |
| --- | --- | --- |
| 系统提示词 | 官方固定提交的结构化原文 | 同原文，仅工具数量及独立 rerank 必要说明 |
| 工具 | search/bash/read 原始说明与共同实现 | 相同三工具，加 v14 独立 rerank |
| 通用反馈 | 原始共同工具返回 | 同 A，不使用 v14 ProgressFeedback |
| 模型 | Qwen3.5-35B-A3B，锁定同一权重、模板与服务 | 同 A |
| 题单 | 原有固定 100 题，原顺序 | 同 A |
| 预算 | 上下文 262144；输出 32768/请求；100 次模型调用；3600 秒/题 | 同 A |
| 采样 | official-thinking-v1；thinking 开启，temperature=1，top_p=.95，top_k=20，min_p=0，presence_penalty=1.5，repetition_penalty=1 | 同 A |
| 检索 | 原始文本 BM25 每子查询 top1000，10×100 字符文件开头预览；原始结构化全文累计导入 | 同 A |
| 截断 | bash 4000 字符，read 默认 2000 行、每行 2000 字符；level0 | 同 A |
| 停止/重试 | 不额外强制终答；API 尝试参数 1，题级尝试 1；自然错误保留 | 同 A |
| 并发 | 两臂共享全局 3，固定配对调度 | 同 A |

rerank 保留 v14 独立服务：累计已导入原文池或显式 file_paths 范围；128-token 段、32-token overlap；默认局部 passage BM25 M1000、神经重排保留 K100、返回 10 段，工具仍允许模型按其原接口调整。Qwen3-Reranker-0.6B revision 固定为 `6e9e69830b95c52b5fd889b7690dda3329508de3`。不自动调用 rerank，不要求 bash→rerank，不改变原文池；没有调用 rerank 也保留为自然结果。

## 官方来源、实际请求与适配

官方源固定为 [texttron/RISE@32672b59](https://github.com/texttron/RISE/tree/32672b59ead8381decae7d412f05bb53d399946a)。提示词/工具的 [逐项差异](prompt_audit/PROMPT_DIFF.md) 包含官方、旧 v14、新 A/B 的完整文本和哈希。A 与官方逐字一致；B 的唯一新增块可逆还原 A。

执行时 `request_contracts/qid_*.json` 在实际发送请求的边界保存系统提示词、工具定义、原始用户问题、配置/契约哈希；每次调用均校验。`api_ledgers` 保存实际采样、预算、请求历史的哈希和 usage，不另抄写推理内容。服务器内原有完整轨迹、断点仅存于 ignored 私有运行目录，不上传 GitHub。

以下是现有本地运行底座相对官方的共同适配，不宣称服务器执行器与论文 macOS 执行器逐字一致。没有在这轮顺带修复工具行为：

1. 官方沙箱使用 macOS sandbox-exec；Linux 使用原有 bubblewrap/Docker 适配，原文工作区只读、网络隔离、每次 bash 使用新 `/tmp`。不延长临时文件寿命。实际后端和 rg 二进制/镜像哈希进入资源清单；不改写 rg 命令或参数。
2. Qwen3.5 通过现有本地 vLLM 适配，原生 reasoning 历史映射、服务端 `/tokenize` 精确预算、同一采样 profile 和完整上下文；服务版本、权重、模板、parser 由实际部署清单锁定。输出预算包括推理 token。原有上下文/输出结束判定保持一致。
3. 现有通用工具含路径范围验证：read 的 `startswith` 路径边界已改为 `is_relative_to`，search 校验合法源文件和导入路径。这是继承的共同安全适配；正常语料路径、内容和截断不变。
4. B 的 search 后回调仅登记已导入文件，返回 `None`；不向 search 返回添加 set ID、池说明或恢复建议。原版运行时错误文字保留。完整源差异见 [adaptations](prompt_audit/adaptations/source_identity.json) 及同目录 `.diff`。
5. 原有逐轮断点与本轮不可覆盖终态用于防止丢进度和结果选择。基础设施中断仅可在同配置/身份/预算的原断点继续，不能从头重跑；自然错误、超时、耗尽、弃答、答错均计入分母。

现有 baseline 的 [复用审计](BASELINE_REUSE_AUDIT.md) 判定旧 100 题运行条件不统一，A/B 均重新生成。旧 v14 61/100 只作为历史整体增强系统结果单列。

## 运行顺序与评分

`runner offline-preflight` 不访问模型。真实资源到位后先 `prepare --config ...` 冻结，`live-preflight` 校验实际模型/262144 tokenizer、共同工具与沙箱、原始 search 返回一致、真实 rerank 可执行，再 `run-all`。

先运行固定前 10 题，共 20 条；只按执行身份、工具契约、模型、token 账本和基础设施健康检查继续，不看正确率。通过后自动续跑余下 90 题，前 10 条原终态直接计入正式结果。每臂 100 题，总共 200 条；配置有任何变化，必须新目录/新清单，不能拼接成绩。

正式评分使用 manifest 预先冻结的统一 rubric，对新公开终答匿名重新评分；不得读取或复用历史 judge/is_correct。评分与每个 final SHA 绑定。报告 A/B 正确率、配对增减和 exact McNemar、轮次/耗时/model token，单列 rerank 调用、段数、输入 token、forward 次数、准备/评分时间与错误。模型输入含缓存，推理是输出子集，均不重复计数；缺失 usage 为未知下界，不记为免费。前 10 题仅检查执行和描述性结果，不据此选择提示词或运行预算。

实际运行资源：按用户选择使用英博云 `q3535bf`，A800 80GB；共同服务为本机 vLLM 0.17.0、BF16、max-num-seqs=3、gpu-memory-utilization=.90，为独立 reranker 留出显存。模型 14 个权重分片和 8 个推理元文件实测匹配官方 revision `59d61f3ce65a6d9863b86d2e96597125219dc754`，见 [来源证明](MODEL_PROVENANCE.json)。原 Docker 镜像 `sha256:4cca46f0b1841ae9708781b564f57bb0f296427b936cf5cd4af143329e0e06b1` 从持久备份恢复，镜像内 rg 15.1.0 字节哈希为 `ebeaf56f8a25e102e9419933423738b3a2a613a444fd749d695e15eba53f71f2`，没有替换二进制或改写命令。

[实际公共配置](yingbo_config.json)在真实容量探针前固定；capacity_report 的路径提前确定，探针报告须匹配配置、源码和稳定服务身份，随后才能 prepare/live-preflight。真实服务进程 PID/启动时钟、命令、模型目录、模板、parser 和关键 vLLM 源码亦随清单核验。完成题目后保存可重建工作区的源文件哈希清单，再释放本轮工作区；原语料、题目轨迹和断点保留。清理只发生在自然终态保存之后，两组一致，不改变题内工具行为。

当前状态：真实容量与执行预检准备；尚未开始生成正式答案。模板中的 `/REQUIRED/` 路径不作为真实资源使用。
