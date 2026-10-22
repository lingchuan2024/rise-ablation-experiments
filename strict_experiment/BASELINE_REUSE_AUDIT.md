# 原版 RISE baseline 复用审计

本轮目标是比较同一套官方结构化 RISE 配置下，增加独立 rerank 工具的增量收益。**现有 Qwen3.5 baseline 不整体复用；A、B 两组重新运行。** 旧 v14 的 61/100 保留为整体增强系统的历史结果，不作为原版 A 组，也不把历史分差解释为 rerank 的单因素收益。

## 官方来源与已匹配项

本地来源记录 `rise-reproduction/LOCAL_RUN_ZH.md` 指向作者仓库 [texttron/RISE](https://github.com/texttron/RISE)，提交 [`32672b59ead8381decae7d412f05bb53d399946a`](https://github.com/texttron/RISE/tree/32672b59ead8381decae7d412f05bb53d399946a)，MIT 许可。官方原版提示词与工具的本轮逐项核验另外记录在提示词合同及差异报告中；不能仅凭目录名字中的“baseline”认定为官方实现。

固定 100 题文件 SHA256 为 `129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c`。旧实验与本轮使用相同题单。`rise-reproduction/SERVER_RUN_ZH.md` 记录：原始 Parquet SHA256 为 `a38ba86c8bbe308cdcc7a067dcec04f15a22a52256afb60d364d04c2fac2f32e`，TOC Parquet 为 `dedd88374e4ed72338817c5ce0a69910783bd38ca7be72fd59405aedfe3c7f44`，文件名映射为 `9061e441070ca95851e13079101cc8388028c6aefaa7b3084da99edb0dc25d1d`。原始文本用于文档 BM25，结构化文本用于工作区；预检仍须验证实际服务器资源，不能只信历史记录。

旧 `head_body_v4_qwen35_20260930/manifest.json` 的 baseline 配置为 `preview_mode=head`，不是 v4 的正文预览臂。历史结构化提示词的 **JSON 编码** SHA256 为 `00f6e5a743c3df2a11b22ce78bdefd667fb28f9a2fae7bb78b19077c945a112a`，工具定义 SHA256 为 `667ffbb62b01abd1bcd2ee6e20b988e062810291d26bd1ecfbe47fa7d857cc11`。JSON 编码哈希与直接 UTF-8 提示词文件哈希是不同口径，不应混用。

## 不满足完整复用的证据

旧 100 题由历史保留的 66 题和健康 Docker 重跑的 34 题组成。重跑 34 题全部替换旧结果，包括退步、弃答和预算耗尽；旧 54/100 保留，合并结果为 52/100。这一结果的保存方式没有取新旧较优，但运行条件不统一。

| 条件 | 历史 baseline | 健康 Docker 重跑 34 题 | 对新实验的含义 |
| --- | --- | --- | --- |
| 模型 | Qwen3.5-35B-A3B，BF16，262144 上下文 | 同名模型、相同精度与上下文 | 仍需锁定本轮实际权重/模板/服务身份 |
| 并发 | 主要与另一臂共享总并发 2 | 4 | 旧 100 题没有统一运行并发 |
| Docker 镜像 | `sha256:57c3e3f1ce320e9ff4397520a47c3cfd38792a6c21d7cecf2bc795acec0e6f45` | `sha256:4cca46f0b1841ae9708781b564f57bb0f296427b936cf5cd4af143329e0e06b1` | 无法作为同一沙箱环境整体复用 |
| 运行状况 | 原 34 题曾出现 393 次 staging-create 故障 | 新 34 题无此故障 | 不能将基础设施失败当模型自然错误，不能任意挑选轨迹 |
| 预算与模型适配 | 100 轮、3600 秒、单次输出 32768、API 重试参数 1 | 保留上述值 | 新两组须统一冻结，而非混用 v14 默认值 |
| v14 历史配置 | 单次输出 32000、API 重试参数 3，另有显存比例/提示词/反馈差异 | 不适用 | 61/100 是不同整体配置的描述性历史结果 |

原始证据（路径相对于研究工作区 `/Users/caozhebo/Documents/科研/9月21日`）：

- `RISE检索空间优化研究/experiments/head_body_v4_qwen35_20260930/manifest.json`：固定题单、baseline 配置与提示词/工具哈希。
- `RISE检索空间优化研究/experiments/baseline_qwen35_docker34_rerun_20261019/README.txt`：66＋34 合并规则、并发与镜像差异、自然错误不得择优替换。
- `analysis/updated_baseline_v14_trajectory_20261019/_work/reports/run_manifest.json`：34 题真实配置、服务身份、提示词与工具哈希。
- `analysis/updated_baseline_v14_trajectory_20261019/REPORT.md` 第 1、9 节：旧对比并非单因素消融，52/100 与 61/100 不能完全归因于 rerank。

## 必须明确记录的共用适配

历史本地 Qwen3.5 服务在英博云 `http://127.0.0.1:8000/v1`，权重目录 `/root/data/q3535bf/model`。这是历史地址；不证明当前 183 服务器已存在同一服务。

1. Qwen3.5 兼容层 `qwen-vllm-v4-native-history` 在请求边界映射原生 `reasoning` 历史，并通过服务端 `/tokenize` 精确计算预算；保留历史而不悄悄截断。旧配置使用 256 token margin、最小有效输出 1024、单请求超时 1200 秒。
2. `official-thinking-v1`：thinking 开启，temperature=1.0、top_p=0.95、top_k=20、min_p=0、presence_penalty=1.5、repetition_penalty=1.0。原论文模型路由不包含此本地 Qwen3.5 服务，属于模型接入适配，A/B 必须相同。
3. 历史 vLLM 0.17.0 使用从官方 0.20.0 回移植的 Qwen3 reasoning parser，处理工具标记隐式结束推理；模板 SHA256 为 `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715`。新环境若版本不同，按实际版本锁定并披露。
4. Linux bubblewrap/Docker、受限工作区与沙箱内 rg 的部署是服务器适配。不得顺便修复 rg 参数、延长 `/tmp` 生命周期、改变 read 行截断或添加额外失败恢复；实际采用后端、二进制/镜像哈希须两组一致。
5. 基础设施暂停、逐轮断点、完整公开输出和 token 审计属于运行保障。恢复时预算连续，不能重新生成已经自然终止的题；两组运行时反馈必须相同。

适配依据：`qwen35_a800_stability_20260930/official_fix_20260930_r2/README.md`、`service_manifest.json`、`code/src/rise/local_vllm.py` 及新实验实际服务预检记录。

## 新 B 组避免混入的 v14 行为

不能直接打开 v14 原 `independent_rerank` 分支：该分支还包裹 `ProgressFeedback.wrap(search/bash/rerank)`，换用 `v13_bash_tool_spec`，并给 search 返回附加工具/池子说明。新 A/B 共用官方 search/bash/read 原始合同与实现；B 仅增加 rerank 注册、必要说明及不可见的累计池登记。提示词的工具数量改动、追加的 rerank 说明和原始共同文字须分别保存、核验。预检不以答题正确率作为继续条件。
