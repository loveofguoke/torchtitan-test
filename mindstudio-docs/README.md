# MindStudio / msProbe 精度调试笔记

本目录整理 MindStudio 官方文档中与本项目相关的 PyTorch 训练精度调试经验。目标不是复制官方手册，而是形成可直接用于 `torchtitan-test` 的选择指南、操作顺序和结果解释规则。

## 版本基线

- 整理日期：2026-09-01
- MindStudio 文档版本：26.1.0
- msProbe 官方仓库快照：`Ascend/msprobe@682d3eb1175031388d033b83c7fd68bb9f23f67d`（2026-08-31）
- 总入口：[MindStudio 文档](https://www.hiascend.com/document/detail/zh/mindstudio/)
- msProbe 官方仓库：[Ascend/msprobe](https://github.com/Ascend/msprobe)

使用前必须再次核对本机的硬件、CANN、PyTorch、`torch_npu` 和 msProbe 版本。本文记录的接口和阈值属于上述版本基线，不能直接视为所有版本或所有模型的验收标准。

## 应该选择哪一种能力

| 问题 | 首选能力 | 本目录文档 |
| --- | --- | --- |
| 同一训练在 GPU 与 NPU 间出现 loss、梯度或中间值差异 | 双端 dump 后执行 `msprobe compare` | [PyTorch 训练精度比对](pytorch-training-accuracy-compare.md) |
| 想快速判断单个 PyTorch API 在 NPU 上是否达到精度标准 | `acc_check`，再用 `api_precision_compare` 比较 NPU/GPU 预检结果 | [PyTorch 离线精度预检](pytorch-offline-accuracy-precheck.md) |
| eager 正常，但 `torch.compile` 后 loss、梯度或输出变化 | `PrecisionChecker` 逐模块比较 eager/compile | [PyTorch 编译精度比对](pytorch-compile-accuracy-compare.md) |

三者不能混用判定门槛：

- 普通训练比对以 dump 数据的余弦、绝对/相对误差、统计量或校验值定位差异。
- 离线预检按数据类型、比较算法和算子看护等级综合判定。
- 编译精度比对当前按 `torch.allclose(atol=1e-4, rtol=1e-3)` 判定逐模块结果。

## 本项目建议的定位顺序

1. 先固定标杆条件：代码提交、checkpoint、输入 batch、随机种子、训练 step、模型模式、数据类型、混合精度配置和并行配置一致。
2. 先做低开销粗定位：使用 `statistics`，优先采集少量 step；需要同时观察 Module 和 API 时使用 `mix`。
3. 从结果中的首个显著差异向前检查输入。不要直接把训练后段大量差异都视为独立根因。
4. 缩小到可疑模块后改用 `tensor` 和更细的 L1/API 范围，计算真实 Tensor 指标。
5. 如果问题只在 `torch.compile` 出现，转用编译精度比对；FSDP2 训练必须使用 single-pass，并保存每个 rank 的独立报告。
6. 将官方阈值作为筛查门槛，再结合本项目已有的 GPU/NPU loss、梯度和收敛验收标准作最终判断。

## 维护约定

- 新增经验时记录：适用场景、硬件与软件版本、复现输入、采集配置、比较命令、指标、结论和官方来源。
- 官方文档与本地安装包行为冲突时，以实际安装版本的 `--help`、源码和最小实验为准，并在笔记中写明差异。
- 不把 `warning` 简化为“通过”，也不把单个启发式阈值越界直接等同于模型最终精度失败。
