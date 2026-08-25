# 图模式调试报告索引

## 汇总文档

- [最终总结](summary.md)
- [代码实现说明](implementation.md)
- [失败与修复历史](failures.md)
- [底层问题定位与提单交接](../../../glm5_2_graph/LOWER_LAYER_ISSUE_HANDOFF.md)
- [完整命令账本](../history/commands.md)
- [主交付报告](../../FINAL_GRAPH_DEBUG_REPORT.md)
- [CANN 9.1.0 安装](../../CANN_9_1_INSTALLATION.md)

## 按卡数与拓扑

| 卡数 | 范围 | 索引 |
|---:|---|---|
| 1 | `single` | [1-card](1-card/index.md) |
| 2 | `ddp2`，以及早期 FSDP2 专用探索 | [2-card](2-card/index.md) |
| 4 | 早期 `ddp4`/`fsdp4` 专用探索 | [4-card](4-card/index.md) |
| 8 | 其余 13 个正式 smoke 拓扑 | [8-card](8-card/index.md) |

正式矩阵共 15 个 topology。4 卡目录记录的是 common 成熟前的扩展调试，不额外增加
正式 smoke topology 数量。

## 最终调用级报告

```text
graph_debug_runs/smoke-suite-inductor-20260825-172213-4113992/reports/report.md
graph_debug_runs/smoke-suite-npugraphs-20260825-172207-4113845/reports/report.md
```

两次调用均发现 15 个已通过的正式 manifest，因此全部跳过并以退出码 0 完成。每个
拓扑的真实训练证据位于对应 `smoke_runs/.../<topology>/manifest.json` 和
`runtime.log`，而不是把这两次快速汇总误当作重新训练了 30 次。
