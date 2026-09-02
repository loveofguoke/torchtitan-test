# MindStudio 工作流服务器验收矩阵

> 本文区分“代码已接入”“官方工具在目标服务器执行成功”“数值结论通过”三种状态。
> 本地单元测试或 CLI 返回 0 不能代替 GPU/NPU 服务器验收，也不能直接写成精度 PASS。

## 1. 状态定义

| 状态 | 含义 | 允许得出的结论 |
| --- | --- | --- |
| `IMPLEMENTED` | adapter、状态机、产物校验和单元测试已实现 | 可以进入服务器最小闭环 |
| `TOOL_READY` | doctor 确认锁定源码、安装包、CLI/import 和运行栈一致 | 当前环境具备运行条件 |
| `CAPTURED` | 子进程成功，官方关键文件非空，manifest/hash/complete 完整 | 该 capture 可供后续官方工具读取 |
| `COMPARED` | 官方 compare/pre-check/visualize 命令成功且输出可解析 | 官方比较阶段完成 |
| `NUMERIC_PASS` | 官方结果的全部相关 forward/backward/rank 状态满足标准 | 仅对本次配置、输入、工具 revision 有效 |
| `TRAINING_PASS` | 目标训练区间的 loss/grad norm、重复运行和稳定性满足标准 | 训练现象通过，不覆盖官方模块/API 失败 |

`CAPTURED` 与 `COMPARED` 是流程状态，不是数值判定。官方 CSV 中未知状态、空结果、
`SKIP` 或项目 parser 无法识别时，必须保持 warning/unparsed，不能按 PASS 处理。

## 2. 正式运行前的版本验收

GPU、NPU 和离线 compare 环境分别执行：

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor --scope accuracy-gpu
python -m tests.glm5_2_mindstudio.toolchain doctor --scope accuracy-npu
python -m tests.glm5_2_mindstudio.toolchain doctor --scope full
```

同时归档仓库外工具目录中的 `toolchain.resolved.json`。正式实验必须满足：

- `toolchain.lock.json` 不再跟随浮动 `master`，而是固定 tag 或完整 commit；
- resolved manifest 中每个 checkout 的实际 HEAD 与 lock 一致；
- GPU/NPU capture 的 msProbe 兼容身份一致；
- NPU 的 driver、firmware、CANN、torch、torch_npu 是官方兼容组合；
- `msprobe` CLI 与 `import msprobe` 指向同一次安装；
- 性能全链路还要求 `msprof`、`msprof-analyze` CLI 与 Python import 同时可用。

## 3. 最小到完整的验收顺序

| 阶段 | 最小命令/范围 | 必查产物 | 通过条件 | 当前代码状态 |
| --- | --- | --- | --- | --- |
| A. fixture | migration `--data --topology single` | token plan、seed checkpoint、generation manifest、两个阶段日志 | 固定输入可重复生成并同步 | `IMPLEMENTED`，待目标环境运行 |
| B. 配置检查 | config-check single 的 reference/candidate/compare | 每 rank 配置包、官方 compare、summary | 两端无未解释的影响精度配置差异 | `IMPLEMENTED`，服务器待验收 |
| C. GPU L0 自检 | 同一 GPU 做两端最小 L0 statistics | `dump.json`、construct、官方 CSV | 工具自身闭环且结果符合预期 | adapter 已实现，需人工建立同设备端点配置验证 |
| D. GPU/NPU L0 | migration single，step 0，rank 0 | 两端 official dump、compare CSV/summary | 所有相关 forward/backward 状态均审阅 | `IMPLEMENTED`，服务器待验收 |
| E. API 预检 | L1 statistics + 两端 `--precheck` + `--precheck-compare` | result/details CSV、precheck report/index | 单端相对 CPU 标杆和两端预检比较均无未解释 error | `IMPLEMENTED`，服务器待验收 |
| F. Monitor V2 | single，100 step，rank 0，weight_grad | `official/rank_0/**/*.csv`、monitor index | CSV 覆盖期望 step，值有限；仅作筛查，不产生跨设备官方 PASS | `IMPLEMENTED`，服务器待验收 |
| G. 分级图可视化 | L0/mix single capture 后 `--graph-visualize` | 非空 construct、`.vis.db`、hash index | TensorBoard Ascend Graph 插件能打开层级图并关联两端 | `IMPLEMENTED`，服务器待验收 |
| H. compile accuracy | NPU single，PrecisionChecker single-pass | 每 rank/part CSV、`compile_coverage_rankN.json`、聚合 CSV、artifact input contract | 每 rank/part 非零 GLM block，存在非 SKIP 的模块 forward/backward 决定性行，不能只有 header/LOSS；runtime input contract 有效；不等价 fullgraph | `IMPLEMENTED`，backend 待验收 |
| I. 代表性分布式 | 至少 DP/FSDP、TP、PP、EP/复合各一例 | 每个选定 rank 的官方结果 | 不丢 rank，PP ownership/part 命名人工确认 | 编排已实现，专项待验收 |
| J. all topology | statistics、窄 step/rank 后再扩大 | suite report、逐 topology summary | 每成员独立完整；失败成员不能被跳过或混入旧 generation | 编排已实现，正式矩阵待运行 |
| K. msProf 系统采集 | profiler-off 基线后，single 与代表性多卡各一份短 capture | `PROF_*`、`msprof_*.db`、runtime/manifest/handoff | doctor `performance-capture` 通过；DB 非空；报告识别 msProf layout | `IMPLEMENTED`，NPU 服务器待验收 |
| L. analyzer 输入矩阵 | msProf 多卡做 cluster；torch_npu capture 做 advisor/compare | 每个 operation status、analysis toolchain、XLSX/HTML/cluster DB | collector/recipe 匹配官方格式；强制重跑不混旧输出 | `IMPLEMENTED`，目标版本待验收 |
| M. Insight 导入 | 完整 single profile、完整多 rank profile、`cluster_analysis_output` | handoff、portable path、GUI 工程/截图或审阅记录 | Summary 后按 Communication/Timeline/Operator/Memory 逐视图确认数据，不把空视图写成通过 | handoff 已实现，26.1 GUI 待验收 |

性能采集和离线分析可以分服务器验收：

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor --scope performance-capture
python -m tests.glm5_2_mindstudio.toolchain doctor --scope performance-torch-npu-capture
python -m tests.glm5_2_mindstudio.toolchain doctor --scope performance-analysis
```

`performance-capture` 验证默认 Ascend PyTorch Profiler；
`performance-torch-npu-capture` 是同一框架级采集环境的兼容 scope；两者都不要求安装
msprof-analyze。`performance-analysis` 不要求 NPU/CANN 运行时。只有需要在同一
环境完成 msProf 采集和离线分析全链路时才用 `--scope performance`。

## 4. Monitor V2 专项矩阵

Monitor V2 首轮只开启 `weight_grad`，确认低开销和 step 计数后再逐项增加：

| 组合 | 需要验证的真实行为 |
| --- | --- |
| single + weight_grad | `patch_optimizer_step=false` 时，每个 TorchTitan train step 只调用一次 `monitor.step()` |
| FSDP/TP/EP + weight_grad | `OptimizersContainer`、DTensor/sharded 参数的梯度统计没有漏参或重复 |
| module + target | 名称过滤命中预期 GLM5 模块，未命中时有明确空结果而非伪 PASS |
| optimizer/param | mv/param distribution 对 sharded optimizer state 的语义符合锁定 msProbe 版本 |
| PP | model-parts 列表、virtual stage 名和 optimizer ownership 正确；当前必须专项验证 |
| compile | 当前 wrapper 明确拒绝 Monitor V2 与正式 `--compile.enable` 组合，不允许静默降级 |

## 5. graph_visualize 专项矩阵

1. reference/candidate 必须来自同一 fixture generation、相同 L0 或 mix 配置；
2. 所选 step/rank 的 `construct.json` 必须存在且非空；
3. 单 rank 先验证，再验证多 rank；
4. `--fuzzy-match`、overflow check、progress log 分别单独 A/B；
5. `.vis.db` 的 SHA-256 和相对路径进入 tracked index；
6. 在服务器启动 TensorBoard，通过 VSCode 端口转发访问 Ascend Graph 插件；
7. Release `analysis` 可同步处理后的 `.vis.db`，但上传前仍需检查模型名、统计量、
   源码/服务器路径；raw tensor 不进入 analysis。

## 6. 数值结论的证据组合

正式迁移建议按以下层次给结论：

```text
配置检查无未解释差异
  + msProbe L0/L1 官方模块/API 比较
  + API pre-check（单端 CPU 标杆 + 两端结论比较）
  + 目标训练区间的 loss/grad norm 和重复运行验收
  + 可疑 API 的单算子复现与修复后 A/B
```

图模式结论独立分层：

```text
PrecisionChecker 被标记模块 single-pass
  + 完整 eager-vs-compile 多 step 精度
  + fullgraph、graph break、guard、recompile、FX/IR/code 诊断
  + profiler 性能 A/B
```

模块/API、训练现象和编译检查是不同证据层，不能互相替代，也不能因某一层通过而
隐藏另一层失败。

## 7. 官方参考

- [MindStudio 文档总入口](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [PyTorch 精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [PyTorch 编译精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md)
- [msProbe Monitor V2](https://mindstudio-docs-master.readthedocs.io/zh-cn/latest/msprobe/docs/zh/user_guide/monitor_v2_instruct/)
- [msProbe PyTorch 快速入门与分级图可视化](https://github.com/Ascend/msprobe/blob/master/docs/en/quick_start/pytorch_quick_start.md)
