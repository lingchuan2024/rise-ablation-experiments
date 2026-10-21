# 官方结构化 RISE：Qwen3.5 严格 rerank 增量实验

本分支的新实验入口是 [strict_experiment/PROTOCOL.md](strict_experiment/PROTOCOL.md)：A 为官方结构化 RISE 原提示词和 search/bash/read，B 只增加独立 rerank 及必要说明。使用同一 Qwen3.5-35B-A3B 和固定 100 题，先运行前 10 题，再原样续跑到 100 题。以下旧 v14 三项实验资料保留为历史，不是本轮的运行配置或对照结果。

- [官方 / v14 / 新 A、B 的提示词逐项差异与完整 diff](strict_experiment/prompt_audit/PROMPT_DIFF.md)
- [旧 baseline 不能直接复用的审计](strict_experiment/BASELINE_REUSE_AUDIT.md)
- [新实验配置、运行与不重跑约束](strict_experiment/PROTOCOL.md)
- [新终答盲评与指标汇总](strict_experiment/ANALYSIS.md)

独立代码及离线检查已完成；用户已选择英博云并启动原 `q3535bf` 实例，Qwen3.5 服务和原 Docker 沙箱已恢复，正在完成真实容量与执行预检，尚无正式答案。[实际英博云配置](strict_experiment/yingbo_config.json)已单独保存。不会使用 Qwen3.8 或旧 v14 结果代替本轮 A/B。

---

# 历史：v14 三项独立消融实验

本目录从冻结 v14 的 `code/` 独立复制，运行三组各100题，共300次问题求解。完整 v14 对照复用历史100题结果；不新增第四组，不混入 v15 bash 修复。历史对照属于既有运行，报告中标注时间和运行环境差异。

| 组别 | 模型可见工具与行为 |
|---|---|
| `search_paging` | search 首屏十篇 snippet，按原检索排名每页十篇翻阅同一次检索的后续结果；保留 bash、read、独立 rerank |
| `auto_rerank` | 每条 search 子查询召回自身 top1000 后立即重排；沿用原子查询，片段 BM25 前1000→神经重排；search 返回前十个完整片段及来源回读坐标；保留 bash/read，不暴露独立 rerank 工具 |
| `no_bash` | search、read、独立 rerank；移除 bash 工具及 bash 指导 |

所有组使用原固定100题、Qwen3.5-35B-A3B、BF16/official-thinking-v1、262144上下文、level0、100轮、3600秒/题、32000单次输出、API重试参数3、read默认2000行、文档BM25 top1000/十篇100字符首预览。固定 `Qwen/Qwen3-Reranker-0.6B@6e9e69830b95c52b5fd889b7690dda3329508de3`，128-token片段、32-token重叠。原始100题文件SHA256为 `129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c`。

`run_ablation.py --ablation-variant ... --space-passage-index ...` 是单组入口。正式套件用 `server/suite.py`：原题单顺序切成十个10题块，每块按 A/B/C、B/C/A、C/A/B 循环运行三组，每次只运行一组、总并发4。首块是健康检查批次，冻结后执行，若源码、参数及资源不变，其30次问题求解直接计入完整300次；后续不重跑这批自然错误。每组共用独立输出根，checkpoint/原文池/重排缓存互相隔离。

## 当前状态

上面的 Qwen3.5/100题方案目前仅完成代码与本地验证。按最新请求实施的 [Qwen3.8 Flash 服务器试跑](HOSTED_PILOT.md) 已于2026-10-21北京时间18:13完成：三个消融各取原题单前10题，共30次求解，总并发3，使用 hosted API。运行代码固定为 `4a754b7d014f0d72004f3480f1da45a3d17107ae`；两套运行配置、清单和结果分别保存。终答、用量及服务器 `completion.json` 已用独立的[本地分析工具](ANALYZE_HOSTED_PILOT.md)核验，[本次试跑结果](results/qwen38_first10_20261021/README.md)保留严格与敏感性口径、逐题哈希和评分标签。

Qwen3.5 正式运行仍需要目标主机上存续的本地模型服务、语料与索引、固定reranker缓存、Docker 环境，以及新的 C4 长上下文与reranker共存容量证据。旧批次容量报告不能作为新服务的通过回执。

## 检查与执行

本地零网络检查：

```bash
python server/suite.py preflight
```

完整本地回归可运行 `python -m pytest -q code/tests --ignore=code/tests/test_v10_cli.py --ignore=code/tests/test_v11_cli.py`。这两个历史CLI测试依赖本目录未交付的旧 `run_v10.py` / `run_v11.py`，与本次入口无关；保留的 `baseline_snapshot` 是真实冻结文件，用于逐字回归核对。

部署后，先使用目标Python在已有服务上运行 `measure-capacity`。它不会启动服务；明确执行代表进行性能请求，消耗与正式300次分开。服务配置JSON须含既有vLLM `command`，launcher回执须含 `pid` 和 `config_sha256`，命令行与当前 `/proc/<pid>/cmdline` 必须匹配。

```bash
python server/suite.py measure-capacity \
  --python /absolute/runtime/bin/python \
  --base-url http://127.0.0.1:8000/v1 \
  --service-config /absolute/service_config.json \
  --service-receipt /absolute/service_launcher.json \
  --hf-home /absolute/hf-cache --reranker-device cuda:0
```

取得本次 `binding.json` 后准备清单，显式提供目标主机真实资源路径：

```bash
python server/suite.py prepare \
  --python /absolute/runtime/bin/python --data-root /absolute/rise-data \
  --base-url http://127.0.0.1:8000/v1 \
  --document-index /absolute/bm25_full --filename-map /absolute/corpus_filename_docid_map.json \
  --corpus /absolute/bcp_plus_structured --passage-index /absolute/passage_index_128_32 \
  --service-config /absolute/service_config.json --service-receipt /absolute/service_launcher.json \
  --capacity-report /absolute/capacity/binding.json \
  --docker-host unix:///absolute/docker.sock --docker-image sha256:EXACT_VERIFIED_IMAGE \
  --hf-home /absolute/hf-cache --reranker-device cuda:0
python server/suite.py preflight --live
python server/suite.py run --first-block-only
python server/suite.py status
python server/suite.py run --resume-after-preflight
```

准备后源码、题单、参数、资源证据及服务身份冻结；不要覆盖清单或已有结果。实时预检验证同服务容量绑定、全部映射来源、真实reranker前向、三波四并发Docker staging/copy/run，以及两次原生模型工具往返。预检成功才可正式运行。基础设施故障停止新问题/工具/模型请求并保留checkpoint；包括native reranker失败，即使工具将异常转换为JSON错误。普通参数错误、自然超时、错误答案和弃答保留为结果。故障修复后必须重新通过 live preflight，再显式恢复，单题原轮次/累计预算继续。

准备和正式状态位于 `server/reports/`；每组结果位于 `runs/<variant>/_per_query/`，完整轨迹在同组 `_traces/`，transport独立累计。各10题块的 `_summary.json` 会被下一块的摘要替换，完整100题分析必须读取 `_per_query/` 与累计transport，不能把最后一块摘要当全组。

## 评估

沿用历史 v14 严格口径，只用公开 final 与冻结gold：无终答和弃答计错，完整姓名和多部分答案必须完整，允许既有等价格式。对新300份公开终答重新复核，不能复用旧脚本中按qid写死的人工判分表。历史参考为 `independent_rerank_v14_20261014/analysis_qwen35_full100_20261019/reports/evaluation.json`（61/100）。报告按组列正确率、逐题转换、轮次、耗时、模型token、工具/翻页次数、自动与主动重排次数及评分开销；容量和预检请求分开计算。历史对照对比不自动构成同期单因素因果证明。

实验结果存在后生成数值汇总与隐藏组别的终答复核包：

```bash
python analyze_ablations.py --output reports/analysis
```

按 `review_packet.json` 复核，填写 `review_template.json` 的 `correct` 和 `reason`，然后重新汇总：

```bash
python analyze_ablations.py --output reports/analysis --reviews reports/analysis/review_template.json
```

只有该组100题全部完成且逐份复核通过才报告完整正确率；未运行、部分完成或未复核时保留相应状态，不把历史标签套用到新终答。代码交付时没有生成实验结果或新准确率。

## 本地验证记录

代码与运行封装回归：304 项通过、2 项跳过。跳过项分别需要显式开启 Docker 集成和 Linux bubblewrap。执行目录为 `code/`：

```bash
python -m pytest -q tests ../server/test_suite.py \
  --ignore=tests/test_v10_cli.py --ignore=tests/test_v11_cli.py \
  --junitxml=../reports/pytest.xml -r s
```

两个排除项依赖本目录不交付的旧 v10/v11 入口；三组新实验的 CLI、工具注册、断点恢复和运行封装均有测试覆盖。`baseline_snapshot/` 仅补齐原 v14 测试缺失的两份真实历史基线文件。

`source_manifest.json` 记录 v14 来源、改动文件、题单和历史评估哈希；`reports/validation.json` 与 `reports/pytest.xml` 保存上述本地交付检查记录。Qwen3.8 试跑另有独立的远程真实预检、启动回执和完成回执，不混入原Qwen3.5验证记录。

## GitHub 版本管理

仓库：[lingchuan2024/rise-ablation-experiments](https://github.com/lingchuan2024/rise-ablation-experiments)，可见性为私有。`main` 保存当前可用代码。

- `v14-frozen-baseline`：原 v14 代码快照，文件哈希见 `baseline_manifest.json`。
- `v14-ablations-v1`：三个消融版本的首个代码交付。
- 后续修改使用功能分支与 Pull Request；正式实验清单记录所运行的 commit，不随运行修改源码。

仅跟踪代码、测试、说明及两份固定研究数据。语料、索引、模型权重、运行轨迹、生成报告和认证信息均由 `.gitignore` 排除。两个固定数据文件通过 `.gitattributes` 保留原始字节，以保持 SHA256 校验。

```bash
git clone https://github.com/lingchuan2024/rise-ablation-experiments.git
cd rise-ablation-experiments
python -m pip install -e code
python server/suite.py preflight
```
