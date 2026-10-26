# RISE：v14 与三组检索消融实验

本仓库整理 Qwen3.5-35B-A3B 上的 v14 完整系统，以及 `auto_rerank`、`search_paging`、`no_bash` 三组变体的求解代码、冻结协议、调度与恢复工具、统一评分入口。

## 已完成结果

2026-10-25 对四组共 320 份公开终答使用同一 DeepSeek V4.1 Flash 规则重新评分；未重新求解，也未沿用旧正确性标签。

| 方法 | 正确 / 实际评分题数 | 正确率 |
|---|---:|---:|
| v14（完整系统） | 61 / 100 | 61% |
| auto_rerank | 68 / 100 | 68% |
| search_paging | 7 / 20 | 35% |
| no_bash | 61 / 100 | 61% |

安全聚合结果见 [RESULTS_20261025.json](evaluation/RESULTS_20261025.json)。分页组只有实际完成的 20 题；相同 20 题上的 v14 为 11/20，不能用它直接比较另一组全 100 题比例。

三组变体由冻结的 **strict B100** 协议派生，v14 是历史完整系统；四组不是所有条件完全相同的单因素实验。自动重排同时改变调用时机、候选范围、返回内容及独立工具可用性，不能把收益全部归于排序模型。自动组另保留 4 题基础设施异常历史，其中 qid310 采用保全的公开终答，不能称其 100 题均为完整自然终态。按预先记录的异常集合排除这 4 题，自动组与 v14 同题为 64/96 和 59/96。无 bash 组完成固定 100 题，保留原读取截断行为，没有额外补偿工具。分页组没有实际翻页调用，结果不能解释为“使用翻页造成下降”。

自动组相对 v14 为 +7 个百分点，配对 McNemar 精确双侧 p=0.189（3 次比较 Holm 校正后 0.568），目前不能据此宣称显著提升。完成数与正确率是不同指标。

## 代码导航

| 目录 / 入口 | 用途 |
|---|---|
| [run_v14.py](run_v14.py) | v14 完整系统启动封装，强制启用独立 rerank |
| [code/](code/) | RISE 求解器、search/read/bash、独立 rerank 与反馈实现 |
| [run_ablation.py](run_ablation.py) | 早期从 v14 派生变体的入口；不是后续 strict 实际执行入口 |
| [strict_experiment/](strict_experiment/) | strict A/B 协议与 B100 对照身份、配置及完整性检查 |
| [strict_ablations/](strict_ablations/README.md) | 三组 strict 工具契约、提示词、预检及调度测试 |
| [operations/](operations/README.md) | 实际续跑、恢复、no_bash 调度、维护、快照与独立验证 |
| [evaluation/](evaluation/README.md) | 四组统一匿名评分与离线汇总工具 |
| [server/](server/) | 历史 v14 服务器部署、预检与运行管理 |

strict 自动组通过 `continuation_suite.py` / `recovery_suite.py` 完成续跑与恢复；分页组使用其独立命名空间；no_bash 使用 `no_bash20_suite.py` 与 `no_bash80_suite.py`，已完成部分只读复用。它们是执行组织方式，不是新增实验组。恢复须保留 checkpoint 和累计预算，不能把 partial 当成新题重置。

## 环境与运行

Python >=3.10。求解器安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e code
python run_v14.py --help
python -m strict_ablations.runner --help
python operations/no_bash80_suite.py --help
```

求解还依赖原始语料、文档 BM25 索引与映射、离线 passage 索引、Qwen3-Reranker-0.6B 权重、Qwen3.5-35B-A3B 模型服务和原隔离运行环境。这些大文件与主机配置不随仓库分发。`run_v14.py` 只启用独立重排，不会自动填入原服务器配置；必须显式指定 `--model`、`--concurrency 4`、索引、语料、输出目录等参数，并使用相应冻结配置的服务环境，不能依赖底层默认模型或并发。

正式 strict 执行使用冻结 common_config：题级并发上限 4、100 turns、3600 秒/题、BF16、262144 上下文、32768 单次输出、API retry=1。历史 v14 的单次输出及重试配置与此不同，具体以各自 manifest 为准。reranker 使用固定版本及 128-token 分段 / 32-token 重叠。不要手抄参数替换 manifest 或为了通过预检修改源码 pins。

依次执行离线检查、资源准备、完整 live-preflight、正式调度。各入口参数及契约见 [strict 文档](strict_ablations/README.md) 和 [operations 文档](operations/README.md)。`live-preflight` 会访问服务；它不是无副作用的离线测试。原 Docker 检查必须继承冻结 `runtime_environment` 中的 `DOCKER_HOST`，不能用默认 socket 的状态替代判断。历史绝对资源路径需在新的实验配置中明确适配；修改后应作为新的执行身份，不能伪装成历史复现。

冻结 solver/tool 的 135 份文件保留原样；本次新增的是便携评分工具与文档。可以先运行不访问模型的检查：

```bash
python -m unittest evaluation.test_uniform_judge -v
python -m unittest discover -s strict_ablations -t .
python -m evaluation.uniform_judge --help
```

## 统一评分

[评分说明](evaluation/README.md) 提供 `collect → prepare → run → summarize` 工作流。只向评分模型提供题目、冻结 gold 与公开终答，去除组名、qid、旧标签与求解轨迹；每题两次独立打乱顺序，出现分歧或不确定标记时新增第三次评审并多数表决。空答为错，题目要求完整姓名时必须满足姓名完整性。评分是对冻结 gold 的一致性判断，不是外部事实核查。

当前结果实际返回模型为 `deepseek-v4-1-flash-261017`。两次评审来自同一模型，仍可能共享系统误差，不能称为人工仲裁或独立模型共识。新评分模块是便携重构版，已离线复核历史 320 份评分并得到一致统计；历史请求仍绑定当时实际执行脚本的 SHA，不能用新代码替代历史运行证明。

旧 [strict_ablations/analyze.py](strict_ablations/analyze.py) 面向 strict control100 + 三组各100 的 **400** 份方案，仍要求完整覆盖；不要为了当前 320 份结果放宽其门禁。

## 数据、隐私与历史记录

`runs/`、`reports/`、工作池、模型、语料、索引、归档与凭据均默认忽略；本次不新增上传轨迹、公开终答集合、评分私有映射或逐题标签。仓库已存在两份冻结研究 fixture：`data/queries_100.jsonl` 和 `data/historical_v14_evaluation.json`，包含基准题目/参考答案及历史评估内容，使用时需遵守相应数据条件。API key 只通过本地环境或隐藏输入提供，禁止写入命令参数、README、配置或 Git。

历史根 README 保留在 [docs/HISTORICAL_README.md](docs/HISTORICAL_README.md)，其“尚未运行”等描述仅对应当时阶段；当前范围与结果以本页为准。上游许可证与来源见 [NOTICE.md](NOTICE.md) 和 [code/LICENSE](code/LICENSE)。

公开前对所有远端分支、标签与 PR 历史进行了凭据检查，范围、误报分类及维护要求见[公开审计记录](docs/PUBLICATION_AUDIT_20261026.md)。
