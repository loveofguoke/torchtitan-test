# 三仓图模式与组合精度实验总结

## 范围

本轮使用当前三仓职责分配：

| 仓库 | 本轮职责 | 基线 |
|---|---|---|
| `torchtitan` | 训练框架与 GLM-5 模型 | `3327058390dad1f48fb61327b66d1bab3deac1a2` |
| `TorchTitanTurbo` | NPU 图模式兼容 patch | `7343c9beda9a067700030a086dc3611ce9db3f54` + 本轮工作区 patch |
| `torchtitan-test` | smoke、combination、运行证据与文档 | `77f4e2eb11e4073155afe38592be9ec4505524aa` + 本轮工作区文档 |

运行栈为 CANN 9.1.0、torch `2.14.0.dev20260805+cpu`、torch_npu `2.14.0`、
triton_ascend `3.2.1`，PyTorch 版本未改动。Python 的 `torchtitan` 与
`torchtitanturbo` 都从上述当前 checkout editable import，不是容器内旧副本。

## smoke 结论

| 后端 | 拓扑 | 实际结果 | 严格含义 |
|---|---:|---|---|
| Inductor | 15/15 | `PASSED` | 原生 Inductor/AOT/Triton，10/10 steps |
| NPUGraphs | 15/15 | `PASSED_AOT_COMPAT` | AOT 执行通过，replay 由兼容 profile 全局关闭 |
| NPUGraphs native replay | 未纳入通过 | 已知失败 | 仍需 torch_npu/图树等底层根治 |

详见 [smoke-graph-validation.md](smoke-graph-validation.md)。

## 本轮新增兼容修复

1. common 把可选 NPUGraph capture mode 导出为空字符串时，Turbo 现在按“未设置”处理，
   避免所有图模式在 import 阶段被空值拒绝。
2. grouped-mm 自定义 backward 的权重梯度改为在 `zeros_like(B_t)` 上执行 in-place
   `index_add_`，确保真实 kernel 与 fake kernel 保持相同 transposed stride，消除 AOT
   `assert_tensor_metadata`。

两项均有 Turbo 单元测试，详见 [failures.md](failures.md)。

## 5000-step 精度

正式 acceptance 是 eager single reference 对 15 个 Inductor candidate topology，固定
step-0 checkpoint/token plan、BF16 mixed precision、seed 61、5000 steps、两个 repeat。
data 已通过，eager reference 的两个 repeat 均完成 5000 steps；single Inductor
candidate 在 deterministic pointwise autotune 处失败，其余 topology 和 compare 未运行。
当前状态为 `BLOCKED`，不是 queued，也不是 precision passed。Turbo/common 已实现仅
pointwise vetted 的 opt-in 兼容，并已通过冷 cache deterministic single 与正式 single
smoke 定向验证，可以恢复正式 candidate 矩阵。只有 `combination
--compare --require-all` 成功后才给出全矩阵通过结论。进度和逐拓扑结果见
[precision-5000.md](precision-5000.md)。

## eager/Inductor 组合性能

固定相同 checkpoint/token plan，完成 single、DDP、FSDP、TP、CP、PP、EP 及复合并行共
15 个拓扑的 eager/Inductor A/B：每种模式两个 repeat，共 60/60 次成功运行。每次运行
30 steps，排除启动/编译 steps 1-10，以 steps 11-30 的中位数比较 steady-state。

6 个含 TP 的拓扑一致获得 1.4552x-1.5400x；不含 TP 的 single/DDP/FSDP/CP/EP 多数在
-1.62% 至 +3.82%，PP-only 为 +5.39% 至 +8.73%。成功的 Inductor 运行没有 graph
break 或 backend failure。由于部分 NPU 为 Alarm，且八卡 Inductor batch 与外部 NPU0
任务重叠，八卡数据是诊断结果，低于 5% 的变化需要在空闲健康节点复验。

完整结果、每拓扑分析、HTML 对比、失败边界和 60 份逐运行过程记录见
[performance/summary.md](performance/summary.md)。

## 产物边界

- Git 内：脚本、`tests/glm5_2_combination/experiments/` 文档与离线汇总工具。
- Git 忽略：smoke/combination 的 runs、trainer output、checkpoint、原始 metrics 和大型
  profiler/compiler 产物；轻量 `*_reports/` 依仓库根目录策略跟踪。
- 仓库外：Inductor/Triton/torch compile cache 与 compiler debug。
- 未执行提交、暂存或推送。
