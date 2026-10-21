# 本地分析Qwen3.8 Flash十题试跑

`analyze_hosted_pilot.py` 只分析下载快照，不修改远程冻结代码，不打开原始trace或checkpoint，不调用模型，也不依据占位 `is_correct` / `judge` 判分。运行中的远程目录不要同步此新脚本：它属于本地分析工具，新增远程根目录源码会改变冻结身份。

快照保留远程相对结构：

```text
reports/hosted_qwen38_first10/
  server/hosted_pilot/reports/run_manifest.json
  server/hosted_pilot/queries_10.jsonl
  runs/hosted_qwen38_first10/<variant>/_per_query/qid_<id>.json
  runs/hosted_qwen38_first10/<variant>/api_transport.jsonl
```

三组必须对应原固定100题的同一前十题。未完成的题保留pending；越界qid、错模型/变体/预算、坏校验和、重复API response_id或不完整JSONL会明确失败，避免混入别批快照或重复核算。会按原canonical算法重算 `run_config_hash`，核对索引/语料/重排资源和128/32切片参数，并拒绝缺失或矛盾的核心计数。活动快照最后一行截断时明确提示重新同步，不静默丢账。可选的 `_hosted_manifest.json` 若存在也会核验。

```bash
python analyze_hosted_pilot.py
python -m unittest discover -s tests -p test_analyze_hosted_pilot.py -v
```

默认输出到快照的 `analysis/`：`metrics.json`、`per_query_metrics.json`、`per_query_metrics.csv`、`report.md`。输出不包含终答全文或原始轨迹，只保留数字指标、qid和公开终答SHA256。

API总量以累计transport为准，同时列出“全部已观察请求”和“已完成qid请求”；前者包含活动题已经完成的API请求。`response_delivered=false` 仍可能有已消耗usage，照常计入。API `prompt_tokens` 已包括缓存；`reasoning_tokens` 是completion子集，不能再加到total。row中的 `agent_usage.input_tokens` 是未缓存输入，与API的inclusive prompt不是同一口径。缺失usage按未知数量标记，不冒充零消耗。

工具、真实检索、子查询数、snippet翻页、主动rerank工具调用、自动rerank子查询、rerank分页和后端评分分别汇总。模型search次数允许多于成功检索和翻页之和，差额单列为失败/未登记调用。A/B的writer最终从manager成功记录重写子查询计数，故会核验与wrapper成功规范化计数相同；C组使用原runner计数，翻页恒为0。模型工具计数包含失败和unknown_tool尝试：C请求bash或B请求rerank会计入 `forbidden_tool_name_attempts`，不会仅凭尝试计数就声称工具实际执行或配置泄露。模型轮次与工具次数分列。题耗时之和、请求耗时之和与观察到的请求时间跨度也分别记录；不把这些量称为正式墙钟。

## 新人工复核

评审文件必须使用本次manifest哈希，并绑定每份公开 `final_text` 的原样UTF-8 SHA256。允许只填已完成子集，但一组十题全部完成且全部新评审前，正式正确率始终为pending。

```json
{
  "version": "hosted-qwen38-analysis-v1",
  "manifest_sha256": "下载的run_manifest中的值",
  "records": [
    {
      "variant": "search_paging",
      "query_id": 116,
      "final_text_sha256": "公开final_text的SHA256",
      "correct": false,
      "reason": "严格口径的新人工判定理由",
      "sensitivity_correct": true
    }
  ]
}
```

`correct` 是主指标。可选 `sensitivity_correct` 单列敏感性汇总：仅对显式提供的记录采用该标签，其余沿用严格标签，不改变主成绩。无终答不能被标为正确。

```bash
python analyze_hosted_pilot.py --reviews reports/hosted_qwen38_first10/fresh_reviews.json
```

## 可选费用估计

没有费率文件时费用为unknown；不会使用结果行里的占位零费用。若提供自行确认的每百万token费率，可计算已知usage估计，仍不代表账单。缺失usage时完整费用保持unknown。

```json
{
  "currency": "自定币种",
  "uncached_input_per_million": 1.0,
  "cached_input_per_million": 0.2,
  "output_per_million": 2.0
}
```

上述数值只是格式示例。使用 `--pricing /absolute/pricing.json` 显式传入实际费率。脚本不联网查价，不比较不同模型历史v14，也不生成历史因果提升结论。
