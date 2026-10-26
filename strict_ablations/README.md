# 从新版 strict B100 派生的三组消融

本目录保留原始 strict 三组各100题的协议与封装。后续实际执行范围为 auto_rerank 100题、search_paging 20题、no_bash 100题；原始三组各100的300次方案未全部完成。当前四组统一评分及代码导航见[根 README](../README.md)，实际续跑/恢复见 [operations](../operations/README.md)。下文“单独试跑20题”及其旧评分是历史阶段记录，不是最新统一评分。离线通过不等于当前服务器已通过 live-preflight。

离线验证共195项通过（新套件110、冻结strict流程75、原工具10）；旧116个冻结源码文件与B100身份完全相同。可查[验证记录](OFFLINE_VERIFICATION.json)及[逐组提示词差异](prompt_audit/PROMPT_DIFF.md)。

## 单独试跑自动重排20题

2026-10-23已完成20题并下载验证全部轨迹，q3535bf已确认关机。40份终答重新匿名评审：自动组13/20、复用control同题11/20；这是探索性历史对照结果，不标记三组正式实验完成。见[报告](AUTO20_REPORT_20261023.md)、[执行与备份回执](AUTO20_EXECUTION_20261023.json)及[完整汇总](AUTO20_RESULTS_20261023.json)。

`python -m strict_ablations.auto20 --root OUTPUT_ROOT` 提供 `template`、`offline-preflight`、`prepare --config CFG --service-pid PID`、`live-preflight`、`run20` 和 `status`。它只运行冻结题单的前20题，使用同一自动重排契约、模型、预算及C4；输出隔离在 `reports/strict_auto20_v1/` 和 `runs/strict_auto20_v1/`，不会调度其他组或control。它是探索性试跑，不标记三组100题实验已完成。

使用 `strict_ablations.auto20_analysis` 的 `export/import/summarize`，将20份新终答与B100同题20份终答组成40份匿名包后统一重新评分。终态与原始请求、账本、trace及checkpoint均须先通过完整性核验。自动重排预检还必须确认真实神经评分的次数和段数均为正。

## 研究对象与对照

对照是2026-10-22完成的**新版 strict B100**：官方结构化RISE提示词，只增加独立rerank接口、参数/返回及必要的`read_args`回读说明。它没有旧v14的bash子集示例、查询优化、数值匹配、长行恢复策略或ProgressFeedback。

复用其固定100题的完整公开终答与数值账本，**不重新运行control100**。新执行的只有三组各100题，共300次；最终统一评分对象是新300份加复用100份，共400份。旧v14整体增强系统、旧Qwen3.8三组10题试跑、分页提示词v2及旧A组中断工件均不是本轮control。

control由固定manifest身份和公开工件哈希确定，不由旧评分确定。关键身份：

- 冻结manifest：`6d64352b47ae0ec09ea2af6763be74d94b451754caf5e096dc98b5c920c67736`。
- control系统提示词：`bb605017783d5c53aa53ba5ac2948842694c15b2f8a7f98f70a4d4d5757293fc`。
- control工具定义：`23b3e8c08214dec94a83f08fcf5f3e4968f4bafdc7c1922eaca231d72599347b`。
- 固定100题：`129a2998335f3b238096c5e2d96a369f1cc690485094cd2df16e203ff90dd67c`。

`control.py`核验完整100题、终答SHA、终态、账本及关机完整性凭据；`prompt.py`另核验control的system/tool/user输入与冻结strict B逐字相同。旧Python模块不作修改，新方法只在`strict_ablations/`命名空间中组合实现。

## 三项干预

| 标识 | 相对control的变化 | 保持的能力 |
|---|---|---|
| `search_paging` | search增加同一次子查询的文档snippet翻页；中性说明调用参数，不强制翻页，也不要求先翻页 | bash、原read、独立rerank |
| `auto_rerank` | search后立即在该子查询自己的候选文档中用原查询重排，返回完整片段；移除独立rerank工具 | bash、原read、累计原文池 |
| `no_bash` | 移除bash工具及不可用能力引用；rerank返回去掉不可执行的`bash_args` | 原search、原read、独立rerank及其所有可用接口说明 |

### 分页组

每条子查询仍按原BM25检索top1000并累计导入完整原文；首屏仍是原始前10篇、每篇100字符开头snippet，仅附加`search_id/offset/next_offset`元数据。后续调用只携带真实`search_id`及可选的零基文档`offset`，不能同时携带新`queries`。每页最多10篇，顺序和snippet口径不变。

翻页不重新检索、不导入文档、不进行神经评分。新的search不使旧文档search记录失效；这与独立rerank的`result_id`及其pool-version失效规则不同。`next_offset=null`只表示该固定记录已到末尾，不能证明全语料无证据。

模型可以自然地选择不翻页。提示词没有“首屏不足时优先翻页”、翻页配额或强制调用流程；翻页率是执行后报告的行为指标，不是筛选运行或调整提示词的条件。

### 自动重排组

一次search中的每条原子查询各自得到top1000文档。每条自动评分只以**该查询自己的文档集合**为范围，不能混入其他查询独有的累计文档；查询文本原样用于局部passage BM25及神经评分。固定请求M1000、K100，返回前10个完整片段，128-token分段、32-token重叠。候选不足时报告实际段数；既有缓存命中不计作新的神经前向。

返回保留真实来源、完整片段、排序信息及`read_args`。没有可调用的独立rerank工具，也不提供隐藏自动排名的`result_id/next_offset`翻页入口。原文仍完整累计导入，bash/read的访问范围不因自动评分改变。自动评分失败不会新增“退回snippet”、自动换查询或额外重试策略。

这项干预同时改变评分时机、查询复用、范围与参数选择、search返回内容及独立工具可用性，研究的是**自动与主动重排的策略联动**。不能把它解释成只开关神经排序模型的单因素实验，也不能把全文片段呈现的收益全部归于神经升排。

### 无bash组

只做能力删除及相应文字清理。`read`仍为原实现、原schema、默认2000行、每行2000字符截断；没有字符切片读取、新恢复端点、路径纠正、额外参数容错或截断补偿。rerank仍返回原有完整片段、来源及可用回读坐标，保留累计池/显式`file_paths`、M/K、结果翻页等说明；长行可能截断的事实不被隐藏。

原strict B已经继承的search参数规范化属于共同底座，各组保持同样行为，不为C增加新容错。也不把共同的结构化文档阅读指导删掉。

## 共同配置、预算与解释范围

`template`直接复制冻结B100的完整`common_config`，`verify_control_bundle(..., common_config=...)`要求所有值和类型相同，包括资源路径。不要凭手工抄写的参数表替换该配置，也不要修改历史服务配置、容量回执或control bundle来通过检查。

共同设置包括Qwen3.5-35B-A3B、BF16、`official-thinking-v1`、262144上下文、每次输出32768、100次模型调用、3600秒/题、`level0`、`api_max_retries=1`、题级尝试1、`coerce_final_on_max_turns=false`、全局并发4；read默认2000行，bash输出截断4000字符。语料、文档BM25、离线passage索引、模型权重/模板/parser、Docker镜像及rg字节身份均需核验。独立rerank固定`Qwen/Qwen3-Reranker-0.6B@6e9e69830b95c52b5fd889b7690dda3329508de3`、128/32分段。

新服务进程属于新的部署epoch；必须显式提供实际PID并记录进程启动身份，不能把历史PID当作当前服务。部署门禁核验静态模型、模板、parser、软件及启动参数只发生允许的生命周期字段变化；同时核对当前GPU索引/型号/显存和Docker镜像，才能复用原C4/长上下文与reranker共存容量证据。本轮不重复长容量探针。历史容量JSON未绑定遥测字节hash、也未记录的旧驱动/UUID/CPU/RAM等，不被声称为已验证完全一致；新epoch记录当前信息和这些证据限制。服务启动、凭据配置和机器管理不由这些命令代办。

三组按原100题顺序展开，共享C4调度。固定前10题×3组的30次pilot只检查执行身份、原生工具往返、真实重排、账本和基础设施健康，不看正确率。配置及源码不变时，30个自然终态直接计入300次，余下270次继续执行，不再重跑前10题。

答错、弃答、空终答、自然时间/轮次/上下文/输出预算耗尽都保留并计入分母。只有基础设施中断可以在通过新预检后从同一checkpoint继续，累计调用和时间预算不重置。已有自然终态不可覆盖，不能为成绩从头重跑。

尽管共同参数及静态资源受控，复用B100仍是**历史对照**；运行时间、服务争用、进程epoch与固定调度可能影响结果。每题一次随机轨迹也不能证明每个答案转换都由干预造成。

## 部署最小公开control bundle

无需复制旧私有轨迹、checkpoint、历史judge或凭据。使用导出命令生成一个此前不存在的目录，只含以下6个原字节文件：

```text
data/queries_100.jsonl
reports/strict_qwen35/run_manifest.json
reports/strict_qwen35/scoring_B_only/public_finals_metrics.json
reports/strict_qwen35/scoring_B_only/frozen_metadata.json
reports/strict_qwen35/operations/b_shutdown_integrity_receipt.json
reports/strict_qwen35/operations/b_completion_check.json
```

导出前后均核验固定身份；输出文件权限为0600，目标目录已存在则拒绝。这些公开证据足以复用终答和已记录数值，不表示中断窗口未落账的用量已知。

## 离线与操作命令

以下命令从包含本目录的**新实验checkout根目录**运行。使用已有依赖齐全的Python解释器，所有路径均为操作时填写的真实路径。`--root`是独立的**工件输出根**，不要求包含`data/`，也不决定Python源码位置；固定题单从源码checkout的冻结`data/queries_100.jsonl`读取。输出根不能指向旧strict或旧v14结果目录。

```bash
ABLATION_PYTHON=/absolute/existing/runtime/bin/python
ABLATION_ROOT=/absolute/new/artifact-root
ABLATION_OLD_CONTROL=/absolute/original/strict-b100-checkout
ABLATION_CONTROL=/absolute/new/b100-public-bundle
ABLATION_CONFIG="$ABLATION_ROOT/strict_ablations_config.json"
mkdir -p "$ABLATION_ROOT"
```

先导出最小control bundle；若已部署相同bundle，则直接用其目录，避免重复导出覆盖：

```bash
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" \
  export-control-bundle --control-root "$ABLATION_OLD_CONTROL" \
  --output "$ABLATION_CONTROL"
```

生成新配置文件并做零模型请求的检查。模板中的`common_config`保持原值；新配置文件选择未占用路径：

```bash
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" \
  template --control-root "$ABLATION_CONTROL" > "$ABLATION_CONFIG"
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" \
  offline-preflight --config "$ABLATION_CONFIG"
"$ABLATION_PYTHON" -m unittest discover -s strict_ablations -p 'test_*.py' -v
```

提示词快照与control相对diff由下列离线命令生成；它不会产生实验答案：

```bash
"$ABLATION_PYTHON" -m strict_ablations.prompt \
  --output-dir "$ABLATION_ROOT/reports/strict_ablations_v1/prompt_audit"
```

真实资源、当前服务身份及容量证据就绪后，才能prepare。`prepare`必须显式提供`--service-pid`，核验并在新输出根保存部署epoch；不要修改旧`service_config`中的PID或原容量回执。

```bash
ABLATION_SERVICE_PID=ACTUAL_CURRENT_SERVICE_PID
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" \
  prepare --config "$ABLATION_CONFIG" --service-pid "$ABLATION_SERVICE_PID"
```

prepare在新路径冻结三组实际prompt/tool/user契约、共同配置、源码、资源身份、control引用、题单和统一评分rubric，拒绝覆盖已有manifest或运行根。之后进行本轮独立live preflight；它会真实调用模型和reranker，探针与正式300次分别记录：

```bash
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" live-preflight
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" run --stage pilot
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" status
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" run --stage full
```

`run --stage full`先核验30次pilot执行凭据，读取已有不可变终态并跳过，不重置自然失败。也可用`run-all`顺序执行pilot和full；不要与上面命令并行启动第二个worker。

基础设施中断后保留原目录，排查并修复外部条件，再显式恢复。下列命令仅适用于原配置、原契约及原checkpoint；若模型服务已重启，向live preflight显式提供新的实际PID，创建允许生命周期变化的新epoch。若服务进程未变，可省略该PID参数：

```bash
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" \
  live-preflight --resume-infrastructure --service-pid "$ABLATION_SERVICE_PID"
"$ABLATION_PYTHON" -m strict_ablations.runner --root "$ABLATION_ROOT" \
  run-all --resume-infrastructure
```

## 新运行工件路径

```text
reports/strict_ablations_v1/run_manifest.json
reports/strict_ablations_v1/preflight_live.json
reports/strict_ablations_v1/pilot_gate.json
reports/strict_ablations_v1/pilot_completion.json
reports/strict_ablations_v1/full_completion.json
runs/strict_ablations_v1/<arm>/_per_query/qid_<id>.json
runs/strict_ablations_v1/<arm>/request_contracts/qid_<id>.json
runs/strict_ablations_v1/<arm>/api_ledgers/qid_<id>.jsonl
runs/strict_ablations_v1/<arm>/_traces/qid_<id>/single.json
runs/strict_ablations_v1/<arm>/_checkpoints/qid_<id>.json
```

三组独立输出、原文池、重排状态及checkpoint；所有模型请求均核验实际发送的系统提示词、工具定义、问题和契约hash。源码、契约或配置变更会使恢复校验失败，不能往旧结果中拼接。私有轨迹及checkpoint继续保存在忽略目录，可由运行门禁本地解析其元数据并核验完整性；**只有公开终答用于评分**，隐藏推理和历史judgement不进入匿名包或判分。

## 400份终答统一fresh盲评

三个新组全部100题自然终止且control bundle完整后，分析器才允许导出400条匿名记录。四组都重新评分，**不能导入旧control的60题标签、旧baseline/v14评分或人工占位字段**。匿名包仅含随机case ID、原题、gold、完整终答及final SHA；arm/qid/terminal SHA映射单独以0600保存。空终答也在包内。

```bash
ABLATION_SCORING="$ABLATION_ROOT/reports/strict_ablations_v1/fresh_scoring"
"$ABLATION_PYTHON" -m strict_ablations.analyze --root "$ABLATION_ROOT" export \
  --out "$ABLATION_SCORING/blind_packet.json" \
  --private-map "$ABLATION_SCORING/private_map.json"
```

评分者在不查看private map的情况下，按导出包冻结rubric逐份比较question/gold/final。提交文件保留包的version、manifest/config/cohort/control/rubric/mapping/packet SHA字段，填写非空`reviewer`，`records`覆盖全部400个case，每条提供原`case_id/final_sha256`、布尔`correct`和非空`reason`。可以将匿名包分给多个审阅者，但导入前需合并为完整400条，不能揭盲后补用历史标签。

```bash
"$ABLATION_PYTHON" -m strict_ablations.analyze --root "$ABLATION_ROOT" import \
  --submission "$ABLATION_SCORING/submission.json" \
  --private-map "$ABLATION_SCORING/private_map.json" \
  --out "$ABLATION_SCORING/fresh_reviews.json"
"$ABLATION_PYTHON" -m strict_ablations.analyze --root "$ABLATION_ROOT" summarize \
  --reviews "$ABLATION_SCORING/fresh_reviews.json" \
  --out "$ABLATION_SCORING/fresh_summary.json"
```

导出、导入、汇总均使用`O_EXCL`创建新文件；已有文件不覆盖。缺题、重复case、不同终答/terminal SHA、旧rubric、部分评分、空答判对都会被拒绝。

汇总分别报告四组正确率、各新组相对control的独对/双对/双错、净增减及两侧exact McNemar。三次比较的p值明确**未经多重比较校正**；不可作为已校正的家族显著性结论。

数值部分包含耗时、轮次、求解模型token、reranker输入/段数/评分与准备时间、主动rerank、自动rerank、文档翻页及重排结果翻页次数。`rerank_calls`为底层服务调用，可能含自动调用和缓存页；不能全当主动神经前向。历史没有记录的检索计数保留null，不填零。模型input包含cached input，reasoning是output子集，不重复相加；未知usage保持未知。重排时间是题内耗时分项，不再加到题耗时上；未配置价格不代表免费。

最终报告应同时给出执行完整性、实际工具使用率和历史对照限制；不会因为开放某项能力就预先声称模型一定使用它或正确率必然提高。
