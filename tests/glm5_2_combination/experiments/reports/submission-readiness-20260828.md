# 图模式与组合性能提交前检查（2026-08-28）

## 结论

Turbo 与 test 的图模式修改是配套修改，建议尽早按两个仓库分别提交并连续推送；不要只推
test 而长期不推 Turbo。当前修改已经具备代码审查条件，但提交说明必须保留以下边界：

- 当前 HEAD 的 Inductor single smoke 和 deterministic cold-cache 10-step 已通过；
- 2026-08-26 基线的 Inductor 15/15 topology smoke 已通过；
- maintained 5000-step eager reference r1/r2 已完成，但 candidate 只在旧实现下运行到
  single-r1 autotune 并失败；采用 G020 workaround 后尚未恢复完整矩阵；
- NPUGraphs 15/15 只代表 replay-disabled AOT-compatible，不代表 native replay；
- 30-step 性能矩阵是 60/60 成功的诊断实验，不是正式精度验收，且健康/争用边界仍在。

因此适合提交的是“兼容实现、运行入口、实验事实与未完成边界”，不适合把提交标题或 PR
写成“全拓扑 5000-step 精度已经通过”或“NPUGraph native replay 已根治”。

## 三仓检查基线

| repository | branch | HEAD | working-tree role |
|---|---|---|---|
| TorchTitan | `feat/glm5-model-distributed` | `59899ade` | clean；device-neutral framework/model |
| TorchTitanTurbo | `glm-dev` | `a5306484` | graph compatibility implementation/tests/docs |
| torchtitan-test | `master` | `01f2f3e1` | common launcher, combination workflow, reports/docs |

运行栈：CANN 9.1.0、torch `2.14.0.dev20260805+cpu`、torch_npu `2.14.0`、
triton-ascend `3.2.1`。TorchTitan 和 Turbo 均由当前 checkout editable import。

## 必须保留的修复

| 修复 | 是否仍需要 | 证据/原因 | 真正根治位置 |
|---|---|---|---|
| 空 NPUGraph capture mode 按 unset 处理 | 是 | clean launcher 会导出空字段；旧 Turbo 在所有 backend import 阶段拒绝 | Turbo typed env contract |
| grouped-mm backward 保持 transposed stride | 是 | fake/real metadata 曾分别为 transposed/contiguous，AOT 校验失败 | Turbo custom-op contract；另向 NPU `index_add` layout 分界 |
| deterministic pointwise autotune vetted | 是 | 正式 5000-step candidate 在 torch_npu benchmark 调用处失败 | torch_npu `NPUCachingAutotuner._bench_with_launch_args` |
| `aten.sum` / functional all-reduce fallback | 是，等待底层分界 | reduction 无合法 UB tile；compiled collective 曾出现数值异常 | torch_npu reduction；PyTorch Inductor collective ↔ HCCL |
| TP `aten.complex` strategy | 是 | 当前 PyTorch DTensor 表仍缺 operator strategy | PyTorch DTensor pointwise registration |
| PP metadata non-batched P2P | 是 | batched object metadata 曾出现损坏 size/EOF | PyTorch PipelineStage ↔ ProcessGroupHCCL |
| EP safe empty grouped-mm | 是 | total/partial empty expert 可触发 zero-core/崩溃 | op-plugin/CANN grouped-mm empty contract |
| zero-numel Triton guard | 是 | 默认 runtime 仍可能 launch grid=0 | torch_npu autotuner/grid policy |
| NPUGraph replay skip | 是，但只能叫降级 | grouped-mm 同步和 `DeviceMesh` runtime input 仍未根治 | op-plugin + torch_npu graph tree/PyTorch AOT |
| task queue/profile/cache/ports/timeouts | 是，属于 launcher policy | 保证环境隔离、可复现和共享机器并发安全 | test common；不进 device-neutral TorchTitan |

完整源码模块、函数和提单验收见
[`LOWER_LAYER_ISSUE_HANDOFF.md`](../../../glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md)。

## 本轮新增验证

### 静态与单元测试

```text
TorchTitanTurbo tests/unit_tests: 43 passed
torchtitan-test tests/unit_tests: 193 passed, 14 skipped
bash -n graph_env_common.sh + run_graph_precision_5000.sh: passed
Turbo/test git diff --check: passed
```

Turbo 单测分别验证 pointwise 传 `is_vetted_benchmarking=True`、reduction 传 false，同时
覆盖空 capture mode、grouped-mm transposed stride，以及 BF16 输入下 NPU Router 的一次
FP32 gate 调用。test 单测覆盖 graph/combination identity、steady-state median、determinism、
smoke contract，并阻止测试顺序造成的“偶然 import torch_npu 后以未打 Turbo patch 的 NPU
执行 parity”环境泄漏。14 个 skip 是没有显式设置 `GLM5_PARITY_DEVICE=npu` 时的预期行为；
正式 parity workflow 会显式传入 endpoint device，因此不受影响。

TorchTitan `ad17686a` 已将通用 Router 从显式 `Linear(compute_dtype=...)` 改为 autocast；
CPU 不支持以 float32 为目标的 autocast。旧 test 仍把 CPU 路径当作 NPU FP32 契约，属于
三仓职责迁移后的测试漂移。本轮把强 FP32 契约落实到 Turbo 单测，test 的 CPU parity
回归只验证上游 module-call/数值契约，没有放松 NPU Router 精度要求。

### 当前 HEAD、cold-cache deterministic single

使用物理 NPU2，独立 cache：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/g020-validation-20260828/
```

命令经 common 调用 `run_graph_mode.sh inductor train`，传入
`--debug.deterministic --training.steps=10` 和 single topology。结果 10/10 steps、退出码 0、
耗时 564 秒；报告明确打印 `Deterministic pointwise autotune is vetted: 1`，且编译日志
实际执行了 `triton_poi_fused_repeat_interleave_select` 等候选 benchmark。

```text
graph_debug_runs/submission-readiness-20260828/
  launcher-train-inductor-20260828-155306-2460699/reports/report.md
```

### 当前 HEAD、正式 smoke runner single

为避免旧 manifest 跳过，用 seed 62 建立新 smoke contract；10/10 steps、退出码 0、耗时
187 秒，started/passed/skipped 为 1/1/0：

```text
graph_debug_runs/submission-readiness-20260828/
  smoke-suite-inductor-20260828-160240-2614125/reports/report.md
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed62-inductor-model/single/
```

两次运行的 logs、trainer output 和 compile debug 都留在 Git 忽略目录；编译 cache 位于
仓库外 `.cache`，没有复制到文档目录。

## 性能实现检查

组合 workflow 的 `--steps` 只允许标识化探索，maintained 默认仍为 5000；
`--performance-nondeterministic` 只允许 performance-only；`--performance-skip-steps`
同时进入新 storage identity 和 steady-state 统计。legacy identity 必须忽略这些后来新增的
字段，以便旧 artifact 仍可迁移/采用，本轮已补回归测试。

现有性能证据为 15 topology × eager/Inductor × 2 repeats = 60/60。统计排除 steps 1-10，
使用 11-30；所有含 TP topology 为 1.4552x-1.5400x，其余大多 -1.62% 至 +3.82%。这些
数据使用 performance-only nondeterministic autotune，不能挪作 precision 结论。详见
[`performance/summary.md`](performance/summary.md)。

## 提交边界

建议 Turbo 和 test 分为两个 commit，顺序为 Turbo 后 test：

1. Turbo：graph compatibility implementation、behavioral tests、patch/dependency/root-fix docs；
2. test：common env、combination workflow/tests、5000-step wrapper、curated experiment docs。

不要把以下内容混入这两个 commit：

- `graph_debug_runs/`、`smoke_runs/`、`combination_runs/`、fixtures/artifacts；
- 仓库外 `.cache`；
- 本任务未审查来源的顶层 `checkpoint_reports/`、`parity_reports/`、
  `performance_reports/`、`stability_reports/`。它们现在允许被 Git 跟踪，但应按各自实验
  单独审查和提交，不能因为 `.gitignore` 策略改变就顺带加入图模式 commit。

推送前仍建议人工查看 staged file list，确保 `tests/glm5_2_combination/experiments/` 中只含
Markdown、一个离线汇总 Python 工具、一个 HTML 和一个 JSON；`__pycache__` 不应进入。
