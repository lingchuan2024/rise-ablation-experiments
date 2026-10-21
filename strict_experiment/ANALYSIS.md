# 严格双臂实验的独立评分与汇总

`analyze.py` 只读取这轮的冻结 manifest、官方题单、公开终答和数值计数。不读取原始轨迹、隐藏推理、历史 v14 成绩或旧 `judge` / `is_correct` 字段。

## 冻结判分规则

runner 在 prepare 阶段把 `rubric_document()` 的 SHA256 写入 `manifest.analysis_contract`。评分规则在正式生成前冻结，不根据分数或某个组的错误调整；终答为空、弃答和自然预算耗尽均计错。对问题明确要求的完整姓名等，要求完整回答；有实质差异的拼写不能自动当作别名。所有歧义按相同规则处理并披露。

单独保存可读规则（新文件；已有文件不覆盖）：

```bash
python -m strict_experiment.analyze rubric --out reports/strict_qwen35/scoring/rubric.json
```

## 新终答盲评

固定前 10 题检查不需要先看正确率。可以等 100 题完成后一次导出全部 200 个新终答；也可以先评分前 10 题，再评分其余 90 题。配置未变时，前 10 题直接保留并计入正式 100 题。

```bash
python -m strict_experiment.analyze export-blind --scope first10 \
  --out reports/strict_qwen35/scoring/first10_blind.json \
  --private-map reports/strict_qwen35/scoring/first10_private_map.json
```

将 `first10_blind.json` 交给独立评审，保留 `version`、`manifest_sha256`、`rubric_sha256`、`mapping_sha256`。评审填写 `reviewer`，为每条记录复制准确的 `case_id` 和 `final_sha256`，增加 Boolean `correct` 和非空 `reason`。不得向评审提供 private map、组名、工具配置、旧答案/旧判分。公开终答自身可能谈到工具，这种文本不能改写，否则不再是实际终答。

导出使用一次生成的随机种子产生稳定匿名 ID 与混合顺序；同一导出的映射哈希固定，私有映射权限 `0600`。不要公开私有种子或映射。重复导出需使用新文件名，不能覆盖原评审。

```bash
python -m strict_experiment.analyze import-reviews --scope first10 \
  --submission reports/strict_qwen35/scoring/first10_submission.json \
  --private-map reports/strict_qwen35/scoring/first10_private_map.json \
  --out reports/strict_qwen35/scoring/first10_reviews.json
```

导入要求每条终答与映射、manifest、配置、提示词/工具合同、行校验和都一致；必须评完该导出的全部案例。旧结果或过期终答的评审不能导入。自然错误不能通过更换终答或重跑来替换。

100 题完成后，可以排除已经评分的这轮前 10 题，只导出剩余终答，随后用同样的方式导入：

```bash
python -m strict_experiment.analyze export-blind --scope all100 \
  --exclude-reviews reports/strict_qwen35/scoring/first10_reviews.json \
  --out reports/strict_qwen35/scoring/remaining90_blind.json \
  --private-map reports/strict_qwen35/scoring/remaining90_private_map.json
```

## 汇总

```bash
python -m strict_experiment.analyze summarize --scope all100 \
  --reviews reports/strict_qwen35/scoring/first10_reviews.json \
  --reviews reports/strict_qwen35/scoring/remaining90_reviews.json \
  --out reports/strict_qwen35/scoring/final_metrics.json
```

默认输入：`reports/strict_qwen35/run_manifest.json`、`data/queries_100.jsonl`、`runs/strict_qwen35/{A,B}/_per_query/qid_*.json`。若在只含公开结果的下载副本分析，可在子命令前指定 `--root` / `--manifest` / `--cohort`。不要求下载任何原始轨迹。

输出包含各组正确数/准确率、终止分布、缺失题、平均/中位轮次与耗时、模型 token、rerank 调用/实际评分次数/片段数/输入 token/评分与准备秒数，以及同题 B−A 差值、四格配对计数和双侧精确 McNemar p 值。全部目标题终止且完成新评分以前，准确率与完整配对检验为 `null`，避免把局部样本当全量成绩。

`input_tokens` 已含缓存输入；缓存只是其中的子集。`output_tokens` 已含 reasoning；不能再次相加。reranker pointwise 输入 token 单列，不能算作模型输出。rerank 的准备和评分秒数已包含于总耗时，不应再加一次。缺少价格时费用保持未知，不把历史占位 `$0` 当免费。

模型 token 与独立数值 API ledger 核账；核账不符拒绝汇总。`api_attempts_without_usage` 非零或缺失时，`api_usage_complete_for_observed_rows=false`，已知 token 总量只是下界，不把未返回 usage 的请求消耗记成 0。

100 题每题只有一次随机轨迹，可能没有足够检验功效；正的分差不是显著提升的证明。旧 v14 只作为不同整体配置的历史背景，不参与这轮 A/B 估计。
