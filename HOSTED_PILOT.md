# Qwen3.8 Flash 三组各十题试跑

这是独立的Hosted API试跑入口，原Qwen3.5完整100题套件保持原配置。三组各运行原固定100题的前十题：70、81、96、116、122、127、132、153、156、161，共30次求解；按 `search_paging`、`auto_rerank`、`no_bash` 依次运行，每组并发3，任一时刻全局最多3题。

求解模型固定请求 `qwen3.8-flash`，端点固定 `https://token-plan.maas.qianwenaiapi.com/compatible-mode/v1`。响应模型默认只接受这个精确名称；如原生探测返回官方快照名，必须在准备前用 `--allow-reported-model EXACT_SNAPSHOT` 明确加入，不允许前缀匹配或静默换模型。

三组统一100轮、累计3600秒/题、read默认2000行、重试参数3。Hosted适配器明确开启 `enable_thinking=true`、`preserve_thinking=true`，推理强度 `medium`。它将原Agent的 `max_tokens=32000` 转为 `max_completion_tokens=32000`，总预算包括推理和答案；不会同时发送两个字段。实际请求形态和账单usage单独写入数值transport，不写密钥或请求消息。恢复时扣除checkpoint中的累计时间，并将剩余时限传递给重试与SDK请求超时。

语料、文档BM25索引、文件映射、128/32片段索引和固定revision的0.6B reranker由目标服务器现有资源提供。支持CPU或明确指定CUDA设备；没有本地vLLM要求，也不会启动或下载求解模型。准备阶段必须提供实际路径，不假定GPU有空闲。三组统一采用 `process_shared_exact_config_serial_score_private_usage_v1`：每个worker进程只加载一份实际reranker权重，各题持有独立用量状态的lease，评分按锁串行，问题/API并发仍为3。live preflight验证三个lease的有限前向、真实sandbox/ripgrep，以及三并发的原生API工具往返；这不是完整负载容量认证。思考模式探针使用 `tool_choice=auto`，不使用该模式不支持的 `required`。

## 运行接口

零网络检查及mock测试：

```bash
python server/hosted_pilot.py preflight
python -m unittest discover -s server -p test_hosted_pilot.py -v
```

离线准备，路径由目标机只读发现结果填入：

```bash
python server/hosted_pilot.py prepare \
  --python /absolute/venv/bin/python \
  --index-dir /absolute/bm25_full --filename-map /absolute/corpus_filename_docid_map.json \
  --corpus /absolute/bcp_plus_structured --passage-index /absolute/passage_index_128_32 \
  --reranker-device cpu --reranker-dtype float32 \
  --hf-home /absolute/hf-cache --sandbox-backend bubblewrap
```

需要CUDA时显式设置 `--visible-devices PHYSICAL_DEVICE --reranker-device cuda:0 --reranker-dtype bfloat16`，先验证该设备允许使用且有足够余量。Docker可用 `--sandbox-backend docker --docker-image EXACT_IMAGE --docker-host unix:///absolute/docker.sock`。

凭据只支持进程环境 `DASHSCOPE_API_KEY` 或 `--secret-stdin` 读取的一行 `{"api_key":"..."}`；不把真实密钥写入命令行、脚本、清单或文件。SSH启动器可以在内存中传递stdin，再由本驱动在内存环境中转交子worker。示例故意不包含任何真实凭据：

```bash
python server/hosted_pilot.py preflight --live --secret-stdin
python server/launch_hosted.py --secret-stdin
python server/hosted_pilot.py status
```

通过live preflight后优先使用 `server/launch_hosted.py` 安全分离启动：它只在内存中把凭据交给子worker，关闭子进程stdin，启动回执不含环境或凭据。直接 `hosted_pilot.py run --secret-stdin` 可用于前台运行；两种入口都受同一套全局锁与冻结清单约束。

准备后源码、资源身份、端点、响应模型白名单及参数冻结。API身份不符、重试耗尽、sandbox/索引/存储/原生reranker故障会停止新问题及新请求，保留checkpoint。修复后重新通过live preflight，再使用 `run --resume-after-preflight --secret-stdin`；自然超时、context limit、达到100轮和错误答案不重跑。

记录在 `server/hosted_pilot/reports/`；三组输出独立位于 `runs/hosted_qwen38_first10/<variant>/`，包含终答、完整轨迹、checkpoint和累计API transport。默认新输出权限仅属主可读。完整轨迹和checkpoint可能包含模型推理，不应提交GitHub。原Qwen3.5分析器不用于此30题Hosted试跑；试跑状态、调用数和usage从本次独立结果统计。
