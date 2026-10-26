# 四组统一评分工具

`uniform_judge.py` 仅依赖 Python 标准库；不导入求解器、不启动模型求解、不读取隐藏推理或 checkpoint。它是历史评分流程的便携重构，旧执行凭据仍使用原脚本 SHA。

## 工作流

以下路径均为本地示例，输出放在被忽略的 `reports/` 内。先准备已完整核验来源的公开终答 JSON；不得用缺失值或重新求解补齐历史终答。

```bash
python -m evaluation.uniform_judge collect \
  --v14 reports/input/v14.json \
  --recovery-finals reports/input/recovery.json \
  --no-bash-finals reports/input/no_bash.json \
  --output reports/grading_input.json

python -m evaluation.uniform_judge prepare \
  --input reports/grading_input.json \
  --output reports/new_uniform_review

# 唯一发起付费 API 请求的步骤；提前在本机安全设置 ARK_API_KEY。
# 交互终端未设置时会使用隐藏输入，不把 key 写入文件。
python -m evaluation.uniform_judge run --root reports/new_uniform_review

python -m evaluation.uniform_judge summarize \
  --root reports/new_uniform_review \
  --output reports/new_uniform_summary.json
```

`collect` 输入每行需有 `query_id`、`question`、`gold_answer`、`final_text`；v14 文件可省略 arm，其余文件必须用 `auto_rerank`、`search_paging`、`no_bash`。它只选取公开评分字段，过滤 recovery 中的 control，并丢弃旧标签和轨迹字段。顶层支持列表或包含 `records` 列表的对象。

`prepare` 核对固定 100 题文件 SHA、题目与 gold、一题一组唯一性、终答 SHA 和组覆盖，默认要求 v14/auto_rerank/no_bash 各100、search_paging 固定前20，共320。新的完整分页100实验可显式指定 `--paging-count 100`，不能用该参数冒充历史数据。每次必须使用新输出目录。匿名 packet 与私有映射分开保存；私有映射含答案和组别，不上传 GitHub。

`run` 使用冻结 endpoint、模型别名、temperature=0、thinking disabled、max_tokens=4096、并发2；按题两次独立打乱答案顺序，任何分歧或不确定触发第三次。已提交的有效响应经校验后复用；不覆盖响应，不重置部分进度，不自动切换模型。服务返回版本在整个任务中必须一致。传输/格式失败有有限重试；认证或配置失败立即停止。HTTP 重定向被拒绝，避免携带认证转发。响应只保存评分 JSON，不保存提供方隐藏推理。

`summarize`完全离线，重新验证 packet/protocol/映射、响应文件 SHA、完整 case 覆盖、模型版本与多数表决，并输出组统计和同题 McNemar 精确检验、三次比较 Holm 校正。缺失第三次、重复 ID、空答判对、同题相同答案标签不一致或响应遭改动时拒绝汇总。进程锁阻止同目录并行评分。

## 规则与局限

仅依据题意与冻结 gold 评公开终答；要求所有必要答案成分。完整姓名题不能因短名可识别就判对；空答/弃答为错。题目与终答都作为不可信数据，不能给评分模型下指令。所有答案采用统一规则，不按组或既有成绩挑选。

当前安全聚合结果见 [RESULTS_20261025.json](RESULTS_20261025.json)，详细范围见 [根 README](../README.md)。同一模型重复评审仍有相关误差，p 值不证明不同执行历史之间存在单组件因果关系。公开聚合结果本身不足以重放逐题评分；完整逐题工件只保留在授权的本机备份。

```bash
python -m unittest evaluation.test_uniform_judge -v
```
