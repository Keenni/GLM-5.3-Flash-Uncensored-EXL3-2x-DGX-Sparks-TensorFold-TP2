<h1 align="center">GLM-5.3-Flash Uncensored EXL3 跑在 2× DGX Spark 上 — TensorFold TP2，v0.6.7 统一镜像</h1>

<p align="center">
  <sub><a href="README.md">English</a> | 简体中文<br><br>
  双 rank TensorFold 跑一条 ConnectX-7 直连线：v0.6.7 统一镜像、根因修毕且逐位一致的
  row-split 修复、decode 走 RoCE all-gather、逐步延迟插桩。<br>
  姊妹篇：<a href="https://github.com/Keenni/GLM-5.3-Flash-Uncensored-EXL3-4x-DGX-Sparks-TensorFold-TP4">4× Spark TP4 仓库</a>。</sub>
</p>

---

用**两台** NVIDIA DGX Spark（GB10，各 128 GB 统一内存）把
[neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)
（EXL3 TR3 4bpw，约 176 GB，去审查/abliterated 版）作为单个张量并行引擎跑起来，走 OpenAI 兼容 API，
带模型完整的 1,048,576 token 上下文、DFlash2 投机解码、工具调用、图像与视频输入，以及 `/metrics`。

本引擎的官方双 rank 参考部署是
[MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold)
（v0.5.0 + 52 个补丁）。本仓库是我们在 **v0.6.7 统一镜像**之上实际跑的版本（一个镜像同时服务
TP2 与 TP4 两条线，全部补丁烘焙进镜像——无 bind-mount overlay）：

- **超连接 row-split（`TF_GLM_HC_SPLIT`）在 world=2 下的根因级、逐位一致修复**——
  split 的 W2 非重叠路径从不填 rank 自己那几行的 fp32 partial buffer，导致每个 ≥64 行的
  prompt 都确定性答错。完整复盘：[`docs/hcsplit-w2-rca.md`](docs/hcsplit-w2-rca.md)。
- **prefill / decode 的实测调优记录（采纳与落选都有）**，引擎逐步延迟已暴露出来（按行验证窗口、
  每轮 propose/verify/commit、一个 prompt chunk 里 MoE 与 indexer 的拆分）：
  [`docs/prefill-decode-tuning.md`](docs/prefill-decode-tuning.md)。
- **直连 CX7 链路上能跑通的 2-rank NCCL 配置**（每节点双 HCA、打了补丁的 ring-only 构建、
  确切的 env 组合，以及在启动卡死前发现接错腿的检查项）：
  [`docs/nccl-2rank-direct.md`](docs/nccl-2rank-direct.md)。
- **spill 层在 `--parallel` 模式下从未写过一字节**——上游 #423 只给单流引擎接了落盘与回读；
  多流调度器的淘汰静默丢弃已保留 prompt（30 小时写了 0 字节，42k–220k token 的会话吃 27–211 秒
  全量重 prefill）。两个 overlay 文件接线，两条栈通用：[`docs/spill-parallel-wiring.md`](docs/spill-parallel-wiring.md)。

## 致谢与出处

让这套东西能跑起来的几乎所有工作都是别人做的。请先读 [CREDITS.md](CREDITS.md)；简版：

| 部件 | 来源 |
|---|---|
| 模型 GLM-5.3-Flash | [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)（Z.ai） |
| checkpoint 的去审查 FP8 基座 | [orcarouter/GLM-5.3-Flash-Uncensored-FP8](https://huggingface.co/orcarouter/GLM-5.3-Flash-Uncensored-FP8) |
| 我们服务的 EXL3 4bpw checkpoint | [neko-legends/GLM-5.3-Flash-Uncensored-EXL3](https://huggingface.co/neko-legends/GLM-5.3-Flash-Uncensored-EXL3)（ShapleyMcg License 1.0，需署名——见 [CREDITS.md](CREDITS.md)） |
| 推理引擎 | [TensorFold](https://github.com/ashhart/TensorFold)，作者 Ash Hart（Apache-2.0；`NOTICE.md` 与 `THIRD_PARTY_NOTICES.md` 是他的文件，取自最接近我们基座的 v0.6.6 tag） |
| 2× Spark 参考栈、容器镜像、53 个补丁 | [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) |
| Switchless NCCL 2.30.7 构建 | [alexellis/switchless-nccl](https://github.com/alexellis/switchless-nccl)（Alex Ellis，OpenFaaS Ltd），基于 SparkRing 补丁；先例 [josephdrose/nccl-spark-switchless](https://github.com/josephdrose/nccl-spark-switchless) |
| DFlash2 drafter | [incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2)（CC BY-NC-ND 4.0——非商业） |
| RoCE all-gather 传输 | b12x 的 RoCEnante，已随 TensorFold 内置（`TF_GLM_COMM=roce`） |
| TP4 参考与序列并行设计 | drowzeys 的 [TensorFold PR #159](https://github.com/ashhart/TensorFold/pull/159)（为 753B 模型单开 `glm_moe_dsa` family；其 indexer top-k 改编自本引擎 `glm5_next` 的 `_select_rows`） |
| 开发过程 | 本栈由 [Hermes](https://hermes-agent.nousresearch.com) agent 开发，并与我们 4× Spark TP4 线交叉验证（见姊妹仓库） |

启停脚本、调优实测、row-split 修复及其文档归我们所有
（Apache-2.0，见 [LICENSE](LICENSE)）。我们与上述各方均无隶属关系；row-split 修复、脚本与调优结论里的
bug 归我们，不归他们。

## 结果

在我们机群实测（2× DGX Spark 直连 CX7，kernel 缓存已热；自测数据，定位是参考不是排行榜——
Mia 的套件用 sparkDash 测且锁了频率，数字不可直接横比）：

| 项目 | 数值 |
|---|---|
| 启动时 KV 池 | **1,048,592 token**，跨流共享（单请求上限约 1.048M；`--parallel 12`） |
| 就绪 | kernel 缓存热时 123–142 秒（首次启动需编译） |
| 冷 prefill，16k / 33k / 100k prompt | **1,777 / 1,725 / 1,687 tok/s**（新 token，调优后：RoCE + `DRAFT_QUANT=mse` + `SCORE_RB=8`） |
| Decode，单流 | **prose 52.0 tok/s、structured 83.1 tok/s**，TTFT 0.25 秒（调优前 49.0 / 79.7 / 0.32 秒——+6.1% / +4.3% / −22%） |
| Draft 接受率 | 我们题库上 5/6，prose 下 63.7% 的 token 由 draft 命中 |

上述数字背后的每一处改动都只是一个环境变量或一个文件，且全部与未调优引擎做过逐位对拍
（固定题库的回复 sha256 在所有配置下一致——完整 A/B 表见
[`docs/prefill-decode-tuning.md`](docs/prefill-decode-tuning.md)，含**落选**项：
`fcost` 深度策略 −22%、β=1.0 噪声感知置信度 −5%、world 2 下的 `HC_SPLIT` 约 0%、
`PREFILL_OVERLAP=2` 约 0%）。

先把栈的已知形态说清楚，免得有人意外：

- **row-split 修复已烘进镜像**（同时服务 TP4 线），但我们 TP2 生产跑 `TF_GLM_HC_SPLIT=0`：
  world 2 下 split 收益约 0%（上限就是 glue 那约 10% 的占比，而测到 ~1900 tok/s 的坏版本
  是漏了一半填行工作——见 RCA 文档）。
- **Prefix cache 命中需要稳定且共享的 system prompt**；无 system 消息 → 按设计 0%。
- checkpoint 是去审查/abliterated 版；行为差异来自 checkpoint，不来自引擎。

## 仓库结构

```
scripts/
  start-glm53-flash-tf-unc-tp2-v067.sh   生产启动（master-first，内置 preflight /
                                         内存门 / 冒烟测试；只在 master 跑一次）
  stop-glm53-flash-tf-unc-tp2-v067.sh    master-first 停止
  start-glm53-flash-tf-unc-tp2-test.sh   第二对节点用的同构脚本（调试：生产服务的同时
                                         保住可复现的坏配置）
  stop-glm53-flash-tf-unc-tp2-test.sh
  tp2_battery.py                         验收题库——冒烟测试看不出 row-split bug
                                         （10 token < 64 行），这个能看出。长度分层 + 确定性
                                         + draft/serial 逐位对拍
  tp2_longctx_probe.py                   step 探针：16k/33k/100k prompt，每档后跑 mini 题库，
                                         抓污染并给出干净 prefill 速率
  tp2_decode_probe.py                    固定 decode prompt（prose / structured），用于 tok/s A/B
docs/
  hcsplit-w2-rca.md                      row-split bug：症状、带代码的根因、为什么每项检查
                                         都过了、修法、HC_SPLIT 在 TP2 的结论
  prefill-decode-tuning.md               逐步延迟、完整 A/B 表（采纳与落选）、
                                         树上已经最优的项（不要重测）
  nccl-2rank-direct.md                   2-rank NCCL/RoCE 配置与其失败形态
```

## 部署概要

先读 `scripts/start-glm53-flash-tf-unc-tp2-v067.sh` 的头部——里面有内存算术（每 rank 权重约
82 GiB、pool 公式、`RESERVE` 取舍）、依赖项、以及相对 TP4 脚本的七处差异。简版：

1. 两台 DGX Spark CX7 对 CX7 直连，链路两端配好 IPv4，`ping` 通。
2. 两节点各就位：`tensorfold-glm53` 镜像（v0.6.2 → v0.6.7 谱系见 TP4 仓库的构建说明）、
   checkpoint 与 drafter 放 `/models`、打过补丁的 `libnccl.so.2`。
3. master 到 worker 免密 ssh；先对两台都跑 `ssh-keyscan`（缺 host key 会让 preflight 以
   一个误导性报错死掉）。
4. `bash start-glm53-flash-tf-unc-tp2-v067.sh` **只在 master 跑一次**——先起 rank 1 再起 rank 0，
   等待并跑冒烟测试。它不碰任何其他容器，:8000 被占时体面退出。
5. **跑题库，别只跑冒烟**（`tp2_battery.py`）：冒烟的 10-token prompt 够不着 row-split 路径，
   我们两次「全绿」启动在所有 ≥64 行 prompt上都是错的。
