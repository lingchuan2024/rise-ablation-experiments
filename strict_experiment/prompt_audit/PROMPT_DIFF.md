# 严格提示词对照：官方原版、现有 v14、新实验 A/B

官方来源：[https://github.com/texttron/RISE@32672b59](https://github.com/texttron/RISE/tree/32672b59ead8381decae7d412f05bb53d399946a)。
源文件以 GitHub Contents API 获取，Git blob SHA 已逐字节核验。

| 维度 | 官方 / 新 A | 现有 v14 | 新 B |
| --- | --- | --- | --- |
| 开场与工具数 | 原文；三种工具 | 改写开场；四种工具 | 原文；仅 three 改为 four |
| search 提示与定义 | 原文 | 系统说明重写 | 与官方逐字相同 |
| bash 提示与定义 | 原文 | 增加长行、数值与路径提示；schema 增 timeout | 与官方逐字相同 |
| read 提示与定义 | 原文 | 改写默认读取策略与长行恢复 | 与官方逐字相同 |
| 结构化文档阅读说明 | 原文 | 复用原文 | 与官方逐字相同 |
| 通用查询优化 / 失败恢复 | 仅原版现有内容 | 增加查询拆分、重复/空结果/截断恢复 | 不增加 |
| bash → rerank 范例 | 无 | 有可选固定范例 | 无 |
| rerank | 无 | 接口及额外范例/策略 | 仅独立接口、参数/返回、read_args 回读说明 |
| 终答和停止说明 | 原文 | 重写 | 与官方逐字相同 |
| 通用运行时反馈 | 官方工具行为 | ProgressFeedback 追加提示 | 与 A 相同；禁用 v14 wrapper |

验证：删除 B 的唯一 rerank 说明块并把 four 还原 three 后，所得字符串与官方 A 完全相同；B 的前三个工具对象与 A 完全相同。
未修正官方系统提示与工具说明中原有的读取偏好差异。

## 完整差异

- [official → existing_v14 system_prompt](official--existing_v14.system_prompt.diff): 有差异，完整补丁已保存
- [official → existing_v14 tools](official--existing_v14.tools.diff): 有差异，完整补丁已保存
- [official → new_A system_prompt](official--new_A.system_prompt.diff): 逐字相同
- [official → new_A tools](official--new_A.tools.diff): 逐字相同
- [official → new_B system_prompt](official--new_B.system_prompt.diff): 有差异，完整补丁已保存
- [official → new_B tools](official--new_B.tools.diff): 有差异，完整补丁已保存
- [new_A → new_B system_prompt](new_A--new_B.system_prompt.diff): 有差异，完整补丁已保存
- [new_A → new_B tools](new_A--new_B.tools.diff): 有差异，完整补丁已保存

## 输入指纹

```json
{
  "version": "official-structured-plus-rerank-minimal-v1",
  "official_repository": "https://github.com/texttron/RISE",
  "official_commit": "32672b59ead8381decae7d412f05bb53d399946a",
  "sources": {
    "rise_agent.py": {
      "git_blob_sha": "99c24fba40002415c9d3fb4011dcebdbcc1fac86",
      "sha256": "b453abd3c9aa2d46f7e28be7a673f8f6c8710cb447424308b722c23f392f07b2"
    },
    "tools.py": {
      "git_blob_sha": "8efb217df87b0611c83ba79fd371101cf96e05ef",
      "sha256": "1754df3bcf6dbaf2cbaba7bdb4ab3ec28fca208b012e520347cfb5496bcbedaf"
    }
  },
  "checks": {
    "official_source_blobs_match": true,
    "A_prompt_exactly_official": true,
    "B_inverse_exactly_A": true,
    "common_tool_definitions_equal": true,
    "common_user_template_equal": true
  },
  "A": {
    "system_prompt_sha256": "98167bd65463d001e0819917363e765a22c9fb842ecf7909d0e9e8d7aef92bc6",
    "tool_specs_sha256": "026693b43a3b9392c153f24127159d0375df759a989d3fa54636cd23f9e5515f",
    "tool_names": [
      "search",
      "bash",
      "read"
    ]
  },
  "B": {
    "system_prompt_sha256": "bb605017783d5c53aa53ba5ac2948842694c15b2f8a7f98f70a4d4d5757293fc",
    "tool_specs_sha256": "23b3e8c08214dec94a83f08fcf5f3e4968f4bafdc7c1922eaca231d72599347b",
    "tool_names": [
      "search",
      "bash",
      "read",
      "rerank"
    ]
  },
  "user_prompt_template_sha256": "6d1d40b7d38fa161ddaa08cf1196e9f529b1c82879c8ede6f20ec95556d61274",
  "rerank_addition_sha256": "8748b4c684c3d585b88b5eb510945acf215ad75d263c501298bd8903ebf24927",
  "runtime_feedback_policy": "official; no v13/v14 observational-feedback wrapper",
  "existing_v14": {
    "system_prompt_sha256": "9c8e97bcc2836371801ab34e2f20bb04a248d76a8281db51e5d0fc67aa980b2a",
    "tool_specs_sha256": "513924447bd517fa54a5baaa4b4ac33a6666ddc2d3f0e975777fcb8fc27f7452"
  }
}
```

这些文件只证明提示词和工具定义的控制；模型、运行配置、数据、工具实际返回与沙箱仍须由运行预检分别核验。
