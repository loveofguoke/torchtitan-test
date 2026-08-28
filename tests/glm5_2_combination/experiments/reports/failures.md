# 组合图模式失败与根治位置

本文件补充图模式总故障档案中本轮实际命中的问题。完整历史仍见
`tests/glm5_2_graph_debug/experiments/reports/failures.md` 和
`implementation.md`。

## C001：空 NPUGraph capture mode 被当成非法值

### 现场

首次 Inductor single smoke 在 import/compat 安装阶段退出。证据：

```text
graph_debug_runs/three-repo-validation-20260826/
  smoke-suite-inductor-20260826-110730-4119670/reports/report.md
```

common 为保证隔离环境字段完整，会导出：

```text
TORCHTITAN_NPUGRAPH_CAPTURE_ERROR_MODE=
```

Turbo 旧逻辑只把变量不存在 (`None`) 当成未设置，空字符串进入枚举校验后被拒绝。即使
当前 backend 是 Inductor，该 compat 初始化也发生在共享 import 阶段，导致所有图模式
失败。

### 本仓方案

`TorchTitanTurbo/torchtitanturbo/tools/graph_compat.py` 的
`_install_npugraph_capture_mode()` 将 `None` 与 `""` 都解释为未设置。单测：

```text
TorchTitanTurbo/tests/unit_tests/test_graph_compat.py::
  test_empty_npugraph_capture_mode_is_unset
```

### 真正根治位置

这是 common launcher 与 Turbo 环境契约问题，应在 Turbo 的环境解析层根治，不应修改
PyTorch、torch_npu 或 CANN。建议长期把可选字符串解析集中为 typed helper，并明确
`unset`/empty/invalid 三种语义，避免每个安装函数各自判断。

## C002：grouped-mm backward fake/real stride 不一致

### 现场

修复 C001 后，single Inductor 在 AOT backward metadata 校验失败。证据：

```text
graph_debug_runs/three-repo-validation-20260826/
  smoke-suite-inductor-20260826-110855-4120250/reports/report.md
```

`safe_grouped_mm_backward` 的 fake kernel 返回 `empty_like(B_t)`，继承 transposed stride
`(65536, 1, 256)`；真实实现使用 out-of-place `torch.index_add(zeros_like(...))`，NPU
结果变成 contiguous stride `(65536, 256, 1)`。AOTAutograd 随后在
`assert_tensor_metadata` 发现 fake/real metadata 不一致。

### 本仓方案

Turbo 新增 `_grouped_mm_weight_grad()`：先 `zeros_like(B_t)`，再原位
`index_add_`，返回值保持 `B_t` 布局。单测：

```text
TorchTitanTurbo/tests/unit_tests/test_graph_compat.py::
  test_grouped_mm_weight_grad_preserves_transposed_layout
```

修复后 Inductor 15/15 拓扑通过，包括 EP8 与 FSDP2+TP4+EP8。

### 真正根治位置

短期归属仍是 Turbo 自定义 op 的 fake/real contract：

```text
TorchTitanTurbo/torchtitanturbo/tools/graph_compat.py
  _install_safe_empty_grouped_mm
  safe_grouped_mm_backward
  safe_grouped_mm_backward_fake
```

更底层应向 PyTorch/torch_npu 提单核查 NPU `aten.index_add` 对非连续 `zeros_like` 输入的
layout 语义是否与 CPU/CUDA 一致，以及 fake tensor 是否应模拟该 layout。候选模块：

```text
PyTorch: torch/_subclasses/fake_tensor.py, torch/_library/fake_impl.py,
         torch/_functorch/_aot_autograd/
torch_npu: aten::index_add 的 NPU lowering/kernel 与 Inductor lowering
```

提单必须附输入 shape/stride、fake 输出 stride、真实输出 stride 和最小复现，不能只附
TorchTitan 训练堆栈。

## C003：NPUGraph native replay 仍未根治

本轮 NPUGraphs 15/15 的结果均由 `TORCHTITAN_NPUGRAPH_SKIP_ALL=1` 获得。当前 workaround
位置：

```text
TorchTitanTurbo/torchtitanturbo/tools/graph_compat.py
  _install_npugraph_skip_policy
torch_npu/utils/_graph_tree.py
  cudagraphify_impl / graph tree capture and replay path
```

已知 native 回归包括 grouped-mm capture error 107030/107027 和 TP `DeviceMesh` graph-tree
输入问题。真正修复需要 torch_npu graph tree 正确处理自定义 grouped-mm、动态/空 expert
shape、非 Tensor 分布式对象输入和 mutation contract；CANN 9.1.0 本身不能解决这些
Python/AOT/graph-tree 语义问题。

在 replay 根治且补充 native smoke/5000-step compare 前，NPUGraphs 结果必须保留
`AOT_COMPAT` 标签。

## C004：deterministic Inductor 拒绝未声明 vetted 的 pointwise autotune

### 现场

正式 5000-step reference 两个 repeat 已完成，candidate 的 `single-r1` 在首个
pointwise Triton kernel benchmark 失败：

```text
graph_debug_runs/precision-5000-20260826/
  launcher-combination-inductor-20260826-211631-1096193/reports/report.md
combination_runs/self-npu-bf16-random-s5000-b64-seq128-seed61-eager-inductor-
  precision-no-prof-skip10-det-447d3961/single/single-r1/runtime.log
```

`torch_npu/_inductor/runtime/triton_heuristics.py::_bench_with_launch_args`
没有向 PyTorch `benchmark_gpu` 传 `is_vetted_benchmarking=True`。PyTorch 正确保留了
deterministic guard，因此这不是通过关闭 deterministic 来修复的精度问题。

### 当前兼容与根治

Turbo 的 `TORCHTITAN_VETTED_POINTWISE_AUTOTUNE=1` 只将
`HeuristicType.POINTWISE` 标记为 vetted；reduction 保持 false。common 对图 profile 默认
启用该 opt-in，并把最终值写入 wrapper 报告。真正根治应落在 torch_npu autotuner，详细
模块、函数、分界与验收见
[G020](../../../glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md#10-g020确定性-pointwise-autotune-未声明-vetted)。

当前结论是 `WORKAROUND_FOCUSED_VALIDATED`：Turbo 单测 6/6、全新 cache 的 deterministic
single 10-step、正式 smoke runner single 10-step 均通过。它仍不等于 5000-step 精度
通过；下一步是恢复 candidate 与严格 compare。
