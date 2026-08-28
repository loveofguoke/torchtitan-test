# 2026-08-26 图模式 smoke 全拓扑验证

## 环境与命令

容器：`glm5-npu-dev`。可见卡：`0,1,2,3,4,5,6,7`。公共环境报告：

```text
graph_debug_runs/three-repo-validation-20260826/
  launcher-env-inductor-20260826-110714-4119609/reports/report.md
```

正式命令：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology all --compiler-diagnostics
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke \
  --topology all --compiler-diagnostics
```

正式 wrapper 报告：

```text
graph_debug_runs/three-repo-validation-20260826/
  smoke-suite-inductor-20260826-111320-4162002/reports/report.md
  smoke-suite-npugraphs-20260826-114532-1009046/reports/report.md
```

## 拓扑矩阵

| topology | cards | Inductor | NPUGraphs profile |
|---|---:|---|---|
| single | 1 | PASS | PASS_AOT_COMPAT |
| ddp2 | 2 | PASS | PASS_AOT_COMPAT |
| ddp8 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp8 | 8 | PASS | PASS_AOT_COMPAT |
| tp8 | 8 | PASS | PASS_AOT_COMPAT |
| cp8 | 8 | PASS | PASS_AOT_COMPAT |
| pp8 | 8 | PASS | PASS_AOT_COMPAT |
| ep8 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp2-tp4 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp2-cp4 | 8 | PASS | PASS_AOT_COMPAT |
| tp2-cp4 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp4-tp2 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp2-pp4 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp2-tp2-pp2 | 8 | PASS | PASS_AOT_COMPAT |
| fsdp2-tp4-ep8 | 8 | PASS | PASS_AOT_COMPAT |

每个 topology 实际完成 10/10 steps、进程退出码 0。PP 的 rank0 日志打印
`loss=-8.00000` 是现有 pipeline 非末级 rank 占位值，不是训练负 loss；combination 的
precision capture 会从正确 endpoint/rank 收集数值，不使用这个占位值做精度判断。

## 代表性 loss

| backend/topology | step 1 | step 10 |
|---|---:|---:|
| Inductor single | 8.13422 | 3.89815 |
| Inductor cp8 | 8.13004 | 3.90827 |
| Inductor ep8 | 8.10556 | 4.03577 |
| Inductor fsdp2-tp4-ep8 | 8.13594 | 3.82363 |
| NPUGraphs single | 8.13422 | 3.89845 |
| NPUGraphs fsdp2-tp4-ep8 | 8.13594 | 3.82463 |

## 为什么 smoke 看起来很慢

训练只有 10 steps，但每个新 topology 会产生不同的分片 shape、collective、pipeline
stage 或 expert layout。Inductor 首次运行需要 Dynamo/AOT tracing、Triton/BiSheng
编译和候选 benchmark；`--compiler-diagnostics` 还会保存 graph/recompile/debug trace。
例如 PP8 首次 8-rank 编译约 2.5 分钟，而进入稳态后约 3.5 秒/step。TP+CP 与
FSDP+TP 组合会为 grouped-mm、scatter、transpose 和通信融合形状重新编译，因此不能
用 10 个稳态 step 的耗时估算整套 smoke。

编译输出均在：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/
  cann91-torch214-triton321/
```

## NPUGraphs 的结论边界

所有 NPUGraphs 日志都明确包含：

```text
NPUGraph replay disabled by compatibility profile
NPUGraph: skipped
```

因此结果分类为 `PASSED_AOT_COMPAT`。它证明 `compile.backend=npugraphs` 的公共启动
接口和 AOT 兼容执行可覆盖 15 拓扑，但不证明 native graph replay 已解决。
