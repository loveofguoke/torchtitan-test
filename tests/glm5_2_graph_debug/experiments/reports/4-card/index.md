# 4-card 图模式探索

正式 smoke 15 topology 中没有独立 `ddp4`/`fsdp4` 名称；本页保留 common 成熟前的
扩展调试记录。

| 编号 | 拓扑 | 结果 | 报告目录 |
|---|---|---|---|
| M04 | DDP4 | 设备释放窗口瞬态失败 | `graph_debug_runs/smoke-ddp-20260824-155129-1406817` |
| M05 | DDP4 | PASSED，245 秒 | `graph_debug_runs/smoke-ddp-20260824-155243-1407460` |
| M06 | FSDP4 | PASSED，99 秒 | `graph_debug_runs/smoke-fsdp-20260824-155654-1424140` |

M04 在物理 NPU5 初始化时报 507033/TsdOpen。设备释放完成后，最小 NPU 张量验证和
原命令重跑均通过，因此分类为瞬态基础设施状态，没有通过修改模型或编译器掩盖。

历史命令：

```bash
ASCEND_RT_VISIBLE_DEVICES=3,4,5,6 GRAPH_NGPU=4 \
  tests/glm5_2_graph_debug/run_npu_inductor.sh smoke-ddp
ASCEND_RT_VISIBLE_DEVICES=3,4,5,6 GRAPH_NGPU=4 \
  tests/glm5_2_graph_debug/run_npu_inductor.sh smoke-fsdp
```
