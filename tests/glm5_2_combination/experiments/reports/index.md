# 组合图模式实验报告索引

| 文档 | 内容 |
|---|---|
| [summary.md](summary.md) | 三仓、smoke、精度实验与结论边界总览 |
| [submission-readiness-20260828.md](submission-readiness-20260828.md) | 当前三仓提交前审计、修复必要性与定向验证 |
| [smoke-graph-validation.md](smoke-graph-validation.md) | Inductor/NPUGraphs 15 拓扑实际运行记录 |
| [precision-5000.md](precision-5000.md) | maintained 5000-step eager/graph 精度矩阵与 acceptance |
| [performance/summary.md](performance/summary.md) | 15 拓扑 eager/Inductor steady-state 性能矩阵 |
| [performance/comparison.html](performance/comparison.html) | 可视化性能对比表 |
| [performance/data.json](performance/data.json) | 60 次运行及 15 拓扑汇总的机器可读数据 |
| [failures.md](failures.md) | 本轮失败现场、临时修复和真正根治位置 |
| [../history/commands.md](../history/commands.md) | 完整命令与断点续跑方法 |
| [../history/performance-commands.md](../history/performance-commands.md) | 组合性能完整执行顺序与复跑命令 |

机器证据不复制到本目录，统一引用 `graph_debug_runs/`、`smoke_runs/`、
`combination_artifacts/` 和 `combination_reports/`。每次组合性能运行的人工可读过程记录位于
`../runs/performance/`，按卡数、拓扑、模式和 repeat 分层。
