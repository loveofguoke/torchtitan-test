# GLM-5.2 图模式调试实验档案

本目录仿照 `tests/glm5_2_performance/explorations/` 组织图模式调试证据。它记录实验
过程和可复现命令，不复制大日志、trainer 输出或编译缓存。

## 目录层级

```text
experiments/
  index.md
  history/
    commands.md              # 实际命令族、最终复现命令、回归开关
  reports/
    index.md                 # 按卡数和拓扑导航
    summary.md               # 最终结论和实现映射
    implementation.md        # 逐文件代码改动、开关和影响范围
    failures.md              # 所有已知失败、根因、修复、复测证据
    1-card/index.md
    2-card/index.md
    4-card/index.md
    8-card/index.md
```

仓库外证据保持原有层级：

```text
graph_debug_runs/<invocation>/logs/runtime.log
graph_debug_runs/<invocation>/reports/report.md
smoke_runs/<contract>/<topology>/manifest.json
smoke_runs/<contract>/<topology>/runtime.log
smoke_runs/<contract>/<topology>/trainer_output/
~/.cache/torchtitan-test/graph_mode/<stack>/
```

## 阅读顺序

1. [总体结论](reports/summary.md)：先确认“跑通”的严格含义和实现位置。
2. [报告索引](reports/index.md)：按 1/2/4/8 卡定位拓扑证据。
3. [代码实现说明](reports/implementation.md)：查看具体文件、默认值和开关。
4. [失败与修复历史](reports/failures.md)：查看每个 bug 的失败现场、根因和复测。
5. [命令账本](history/commands.md)：复现最终配置或关闭单个兼容项做回归。

## 证据优先级

1. 每个正式 topology 的 `manifest.json` 是该 topology 是否完成的机器可读依据。
2. `graph_debug_runs/.../reports/report.md` 记录一次 wrapper 调用的环境、命令和分类。
3. `runtime.log` 用于错误定位和训练 step/loss 证据。
4. 本目录 Markdown 是索引和解释，不覆盖上述机器证据。

最终 Inductor 结论要求真实编译训练完成 10/10 steps、退出码 0、数值检查无异常。
NPUGraphs profile 采用 `TORCHTITAN_NPUGRAPH_SKIP_ALL=1`，其通过只表示 AOT 兼容
降级，不表示 NPUGraph 原生 capture/replay 通过。
