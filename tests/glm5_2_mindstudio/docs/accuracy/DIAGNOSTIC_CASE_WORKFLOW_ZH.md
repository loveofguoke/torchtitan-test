# GLM-5.2 MindStudio 精度诊断 Case 工作流

`accuracy_diagnostic_benchmark.py` 在现有官方 msProbe 实验之上提供有状态精度诊断控制面。它不
重新实现 ConfigChecker、PrecisionDebugger、compare、Monitor V2、Trend Analyzer、
graph_visualize、overflow check 或 API 预检，也不复制官方产物。它解决的是原来
缺少的编排问题：面对一个具体精度现象，下一步应该运行什么、依据什么进入下一阶段、
证据在哪里，以及修复是否真正闭环。

官方方法依据是[大模型训练精度定位指南](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/LargeModelTrainingAccuracy/docs/zh/best_practices/train_debug_guide.md)。官方给出问题线和定位方法，但没有规定所有模型统一的长程步数、Grad Norm 阈值和任务指标容差。因此 recipe 中的 `100` 个 monitor step 只是醒目的占位值，执行前必须根据问题复现窗口修改，不能作为官方验收标准。

## 1. 建立 Case

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py init glm5-first-loss-001 \
  --title "GLM-5 GPU/NPU first-step Loss difference" \
  --symptom first-step-loss \
  --topology fsdp8 \
  --notes "Loss mean relative error first exceeds the project gate at step 1."
```

可选现象为：

- `nan-or-overflow`：NPU 比标杆更早或更频繁出现 NaN/Inf；
- `first-step-loss`：第一步或前几步 Loss 已异常；
- `long-term-loss`：前期拟合、后期持续偏离；
- `spike`：Loss/Grad Norm 尖刺；
- `downstream-metric`：Loss 接近但最终任务指标下降；
- `unstable`：固定随机性后仍不能稳定复现；
- `unknown`：先用 Monitor/Trend 观察后再人工分类。

输出位于：

```text
mindstudio_cases/accuracy/<case-id>/
├── case.json       # 唯一状态源
├── next_plan.json  # 当前阶段的机器可读 recipe
└── README.md       # 阶段、证据、第一现场、假设和下一命令
```

创建后会立即输出 CheckList recipe。命令是现有 benchmark 的正常命令，因此继续
遵守原来的 fixture generation、断点重跑、`--force` 和官方产物目录契约。GPU 与
NPU 命令仍在各自服务器执行；case 只在证据汇合的位置维护。

## 2. 六道门

```text
checklist -> reproduce -> observe -> localize -> verify -> validate -> close
```

1. `checklist`：ConfigChecker 之外还要附上模型结构、checkpoint/token hash、源码和 rank 映射。只有所有影响精度的差异都消除或解释后才能记 `pass`。
2. `reproduce`：同端重复，结论为 `stable`、`unstable`、`not-reproduced` 或 `inconclusive`。跨端差异不能代替同端复现。
3. `observe`：用整网 Loss、Grad Norm、NaN/Inf、尖刺和任务指标证明现象 `normal` 或 `abnormal`，并确定第一异常窗口。
4. `localize`：按现象选择 L0/mix、MD5、overflow、Monitor/Trend 或 L1，下钻到 step、rank、phase 和 module/API。
5. `verify`：使用真实 tensor、API 预检以及 FP32、单 API 移 CPU、融合拆解等单变量 A/B，证明嫌疑点是误差源而非受害者。
6. `validate`：修复后依次验证局部 API、原第一现场、长程、目标拓扑和最终任务指标。

每次新的 MindStudio capture 还会在对应 run 根目录写入
`training_metrics.jsonl`。这是 TorchTitan 日志点的整网 Loss、global max Loss 和
Grad Norm，不属于也不会混入 msProbe official 目录。它保留完整数值以及
`NaN/Infinity`，用于确认整网现象；msProbe official 数据继续负责模块/API 下钻。

`reproduce` recipe 会保留 migration 的 `candidate-r1` 和 `candidate-r2` 两份 MD5
capture，再通过 `accuracy_diagnostic_benchmark.py compare-repeats` 调用官方 `msprobe
compare`。结果位于 case 的 `02_reproduce/candidate-r1-vs-r2/`；重复执行时，输入
manifest 未变化就跳过，失败产物则归档后重试。原有 r1 不会为了复现检查被覆盖。

阶段必须按顺序完成，且每个阶段至少引用一个真实存在的证据文件或目录：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py record glm5-first-loss-001 \
  --stage checklist \
  --conclusion pass \
  --evidence mindstudio_reports/accuracy/<config-check-id> \
  --evidence mindstudio_fixtures/accuracy/<migration-id>/fixture.json \
  --notes "Reviewed all rank sheets; CUDA/NPU-only packages are expected differences."
```

以 `unknown` 建立的 case 在 `observe` 时应通过同一条 `record` 命令增加
`--symptom long-term-loss` 等参数完成分类；后续 recipe 会立即按新现象切换。

命令结束会自动刷新 `case.json`、`next_plan.json` 和 `README.md`，并打印下一阶段
recipe。随时可以查看：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py status glm5-first-loss-001
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py plan glm5-first-loss-001
```

### 2.1 整网曲线和现象摘要

完成 reference/candidate capture 后，当前 `observe` recipe 会自动给出：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py training-observation \
  glm5-first-loss-001 --workflow migration
```

长稳和尖刺场景使用 `--workflow monitor`。每个拓扑生成：

```text
mindstudio_cases/accuracy/<case-id>/03_observe/<workflow>/<topology>/
├── training_metrics_compare.csv
├── summary.json
├── loss.svg
├── grad_norm.svg
└── relative_error.svg
```

`summary.json` 给出双方首个非有限值、Loss 首个超过指导阈值的 step、平均相对误差
和可选尖刺位置。Loss 默认 `1%` 只用于复现官方案例中的现象分类，不会自动把整个
实验判为通过或失败。官方没有给出统一 Grad Norm 和尖刺阈值，因此默认只画曲线；
需要项目阈值时显式传入：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py training-observation \
  glm5-first-loss-001 --workflow monitor \
  --grad-norm-relative-threshold 0.05 \
  --spike-relative-threshold 0.20
```

输入 hash 和阈值未变化时会断点跳过。改变输入或阈值必须使用 `--force`，且只归档
并替换 Case 内的派生图表，不会删除 run、artifact 或 msProbe official 数据。旧的
已完成 capture 若没有 `training_metrics.jsonl`，需要重新 capture 才能生成精确曲线；
不从终端四舍五入后的日志猜测原始数值。

## 3. 登记第一现场

在 `observe` 阶段附上整网证据；到 `localize` 时登记明确位置：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py record glm5-first-loss-001 \
  --stage localize \
  --conclusion localized \
  --step 1 \
  --rank 3 \
  --phase backward \
  --module Module.0.layers.5.attention \
  --api Torch.matmul.83.backward \
  --evidence mindstudio_reports/accuracy/<migration-id>/fsdp8/official_compare \
  --notes "The input is aligned; the first material output divergence is here."
```

下一条 recipe 会把该 step/rank/API 带入 L1 tensor、compare 和 pre-check 命令。
如果只知道 module，可以不填 API；如果连 module 都不知道，`localized` 不应通过。

## 4. 用假设而不是猜测管理根因

每个候选解释必须声明一个能够证伪的单变量实验：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py hypothesis add \
  glm5-first-loss-001 \
  --statement "The fused attention mask contract causes the first divergence" \
  --experiment "Keep inputs fixed and replace only fused attention with the reference decomposition."
```

实验完成后记录支持、否定、部分支持或无法判断：

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py hypothesis verdict \
  glm5-first-loss-001 1 \
  --verdict supported \
  --evidence mindstudio_reports/accuracy/<ab-id> \
  --notes "The local tensor and original Loss window both align after the one-variable change."
```

`verify=confirmed` 仍不能直接关闭问题；必须完成修复后的长程、拓扑和任务验证。

## 5. 关闭条件

```bash
python tests/glm5_2_mindstudio/accuracy_diagnostic_benchmark.py close glm5-first-loss-001
```

关闭会强制检查：六个阶段全部有证据；CheckList 为 `pass`；整网现象确认为
`abnormal`；第一现场为 `localized`；根因为 `confirmed`；至少一个假设得到
`supported`；最终验证为 `pass`。缺少任何一项都会拒绝关闭，而不是把工具运行成功
误报为精度通过。

## 6. 当前边界

该控制面已经覆盖预训练迁移中的 CheckList、稳定/不稳定复现、NaN、首步、长稳、
尖刺、下游异常、模块/API 下钻、预检和修复回归的状态组织。它不会自动替人判断
一条差异是否合理，也不会虚构跨设备 Monitor PASS。

官方强化学习案例中的 rollout、reward、actor/rollout resharding、训推一致性和
KV Cache 需要独立推理与 RL fixture，目前不属于这个预训练 case 的已验证范围。
内存踩踏指向 kernel 后的 pointer/stream 日志与 msSanitizer 也属于专项工具，case
可以引用其证据，但当前不会自动生成侵入式补丁。
