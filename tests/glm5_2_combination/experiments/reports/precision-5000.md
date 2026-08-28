# eager/Inductor 5000-step 组合精度报告

## 正式配置

| 字段 | 值 |
|---|---|
| reference | NPU single eager |
| candidate | NPU selected topology, Inductor model compile |
| steps | 5000 |
| repeats | 2 |
| seed | 61 |
| sequence length | 128 |
| local/global batch | 8 / 64 |
| training dtype | FP32 |
| parameter dtype | BF16 |
| reduce dtype | FP32 |
| deterministic | true |
| checkpoint/input | shared step-0 checkpoint + fixed token plan |
| acceptance | `--compare --require-all` |

正式入口：

```bash
bash tests/glm5_2_combination/run_graph_precision_5000.sh inductor all
```

## 本轮执行状态

状态：`BLOCKED`（2026-08-28 核查）。2026-08-26 的持久任务已经结束，不再有
`graph_precision_5000_20260826` tmux 会话。实际完成情况是：

1. fixture/data 阶段通过，耗时 122 秒；
2. eager reference 的 r1/r2 均完成 5000 steps，共耗时 30,822 秒（约 8.56 小时）；
3. Inductor candidate 在 `single-r1` 第一次 pointwise Triton autotune 时失败，耗时 41 秒；
4. 其余 candidate topology 与最终 `compare --require-all` 均未运行。

wrapper 报告位于：

```text
graph_debug_runs/precision-5000-20260826/
  launcher-combination-inductor-20260826-124047-1095454/reports/report.md  # data PASS
  launcher-combination-inductor-20260826-124249-1096458/reports/report.md  # reference PASS
  launcher-combination-inductor-20260826-211631-1096193/reports/report.md  # candidate FAIL
```

eager artifacts 已经完整保存；r1/r2 的 `metrics.jsonl` 均恰好 5000 行，step 5000 的
loss/grad norm 同为 `2.230001449584961` / `0.23182907700538635`。这证明 reference 与固定
输入契约可复用，但没有 candidate 结果，不能得出 eager/graph 精度结论。

失败根因不是训练慢或 CANN 版本污染。torch_npu 的
`NPUCachingAutotuner._bench_with_launch_args` 调用 PyTorch
`benchmark_gpu(..., device_type="npu")` 时没有把 pointwise 场景标记为
`is_vetted_benchmarking=True`，于是正式精度的 deterministic guard 主动拒绝 benchmark。
完整定位和真正 patch 位置见
[G020](../../../glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md#10-g020确定性-pointwise-autotune-未声明-vetted)。

Turbo 已增加 opt-in、仅 pointwise 生效的兼容实现，common 默认通过
`GRAPH_VETTED_POINTWISE_AUTOTUNE=1` 选择它；reduction 仍不允许在 deterministic mode
benchmark。2026-08-28 已用全新仓库外 cache 完成 deterministic single 10-step 定向验证，
并另用正式 smoke runner 完成 single 10-step；这允许恢复 candidate，但不替代 5000-step
精度验收：

```bash
bash tests/glm5_2_combination/run_graph_precision_5000.sh inductor candidate
bash tests/glm5_2_combination/run_graph_precision_5000.sh inductor compare
```

恢复时已有合法 fixture/reference artifact 会被自动跳过。不得增加
`--performance-nondeterministic`：该选项只属于性能探索，会破坏正式精度的确定性验收。
5000 steps 不是 smoke；15 个 candidate topology 各有两个 repeat，各并行布局不能共享
全部编译图，完整任务仍预期以天计。最终 compare 成功前，本报告不声明精度通过。

复用旧 reference 不是无条件规则。本次源码差异审计确认：TorchTitan 从该运行基线到
`59899ade` 只修改 GLM MFU/FLOPs 估算，Turbo 后续提交没有修改 eager GLM 数学，test 的
变化属于生命周期、报告和图兼容入口，因此现有 loss/grad-norm reference 仍可作为本轮
续跑输入；报告中的三仓 source metadata 必须保留。以后只要 GLM forward/backward、
router、optimizer、数据顺序、训练参数或 NPU eager patch 任一项变化，就必须重新执行
`--data --force` 和 eager reference，不能沿用本条结论。

## 进度与结果

本表只依据 `combination_artifacts` 和最终 compare 报告更新；未运行完成的项不得标为
PASS。

| topology | repeat 1 | repeat 2 | eager/Inductor compare |
|---|---|---|---|
| single | FAILED AT AUTOTUNE | NOT RUN | BLOCKED |
| ddp2 | NOT RUN | NOT RUN | BLOCKED |
| ddp8 | NOT RUN | NOT RUN | BLOCKED |
| fsdp8 | NOT RUN | NOT RUN | BLOCKED |
| tp8 | NOT RUN | NOT RUN | BLOCKED |
| cp8 | NOT RUN | NOT RUN | BLOCKED |
| pp8 | NOT RUN | NOT RUN | BLOCKED |
| ep8 | NOT RUN | NOT RUN | BLOCKED |
| fsdp2-tp4 | NOT RUN | NOT RUN | BLOCKED |
| fsdp2-cp4 | NOT RUN | NOT RUN | BLOCKED |
| tp2-cp4 | NOT RUN | NOT RUN | BLOCKED |
| fsdp4-tp2 | NOT RUN | NOT RUN | BLOCKED |
| fsdp2-pp4 | NOT RUN | NOT RUN | BLOCKED |
| fsdp2-tp2-pp2 | NOT RUN | NOT RUN | BLOCKED |
| fsdp2-tp4-ep8 | NOT RUN | NOT RUN | BLOCKED |

## 报告判读

compare 必须同时验证完整 step 序列、有限值、固定输入契约和 repeat 完整性。任何 topology
缺少 endpoint artifact、训练提前退出或超过 maintained tolerance 时，`--require-all`
应失败；不能通过放宽阈值或删掉失败 topology 得到“全通过”。

运行目录：

```text
precision_fixtures/self-npu-bf16-random-s5000-b64-seq128-seed61-*/
combination_runs/self-npu-bf16-random-s5000-b64-seq128-seed61-eager-inductor-precision-*/
combination_artifacts/self-npu-bf16-random-s5000-b64-seq128-seed61-eager-inductor-precision-*/
combination_reports/self-npu-bf16-random-s5000-b64-seq128-seed61-eager-inductor-precision-*/
```

`fixtures`、`runs` 和 `artifacts` 是 Git 忽略的运行证据；轻量
`combination_reports/` 按仓库根目录策略可进入 Git。本文件记录历史运行状态和相对路径，
不复制原始日志或 checkpoint。

2026-08-28 定向验证报告：

```text
graph_debug_runs/submission-readiness-20260828/
  launcher-train-inductor-20260828-155306-2460699/reports/report.md
  smoke-suite-inductor-20260828-160240-2614125/reports/report.md
```
