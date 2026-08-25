# 1-card 图模式实验

## single

| 后端 | 结果 | 训练证据 | 调用报告 |
|---|---|---|---|
| Inductor | PASSED，10/10 | `smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-inductor-model/single/manifest.json` | `graph_debug_runs/smoke-suite-inductor-20260824-174917-1691961/reports/report.md` |
| NPUGraphs profile | PASSED，AOT 降级 | `smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-npugraphs-model/single/manifest.json` | `graph_debug_runs/smoke-suite-npugraphs-20260824-181603-1824452/reports/report.md` |

复现命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke --topology single
```

single 首先验证了 CANN 9.1 header/PCH、Inductor reduction fallback 和 grouped-mm
NPUGraph capture 边界。NPUGraphs 行不能解读为 native replay 通过。
