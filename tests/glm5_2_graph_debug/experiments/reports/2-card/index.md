# 2-card 图模式实验

## 正式 ddp2

| 后端 | 结果 | 训练证据 |
|---|---|---|
| Inductor | PASSED，10/10 | `smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-inductor-model/ddp2/manifest.json` |
| NPUGraphs profile | PASSED，AOT 降级 | `smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-npugraphs-model/ddp2/manifest.json` |

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology ddp2
ASCEND_RT_VISIBLE_DEVICES=0,1 \
  tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke --topology ddp2
```

## 早期专用探索

common smoke matrix 成熟前还运行过 `run_npu_inductor.sh smoke-ddp` 和
`smoke-fsdp` 的 2 卡路径：

| 编号 | 路径 | 结果 |
|---|---|---|
| M01 | `graph_debug_runs/smoke-ddp-20260824-154206-976772` | 外部资源冲突，中止 |
| M02 | `graph_debug_runs/smoke-ddp-20260824-154626-1178208` | PASSED，224 秒 |
| M03 | `graph_debug_runs/smoke-fsdp-20260824-155016-1397416` | PASSED，64 秒 |

早期 FSDP2 是调试证据，不是当前 15 topology 正式矩阵中的独立名称。
