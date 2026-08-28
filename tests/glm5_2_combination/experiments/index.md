# GLM-5.2 组合图模式实验档案

本目录记录三仓结构下图模式 smoke、eager/graph 组合精度与组合性能实验。组织方式参考
`tests/glm5_2_graph_debug/experiments/` 和
`tests/glm5_2_performance/explorations/`，只提交可复现文档，不提交大日志、checkpoint、
TensorBoard 文件或编译缓存。

## 目录层级

```text
experiments/
  index.md
  history/
    commands.md
    performance-commands.md
  reports/
    index.md
    summary.md
    smoke-graph-validation.md
    precision-5000.md
    failures.md
    performance/
      summary.md
      comparison.html
      data.json
      1-card/
      2-card/
      8-card/
  runs/
    performance/<card-scope>/<topology>/<mode-repeat>/readme.md
  tools/
    summarize_performance.py
```

运行证据按仓库统一生命周期保存：

```text
graph_debug_runs/<invocation>/logs/runtime.log
graph_debug_runs/<invocation>/reports/report.md
smoke_runs/<contract>/<topology>/
precision_fixtures/<fixture>/
combination_runs/<experiment>/<topology>/<endpoint-repeat>/
combination_artifacts/<experiment>/<topology>/<endpoint-repeat>/
combination_reports/<experiment>/
```

其中 fixtures、runs、artifacts、`graph_debug_runs` 和编译缓存被 Git 忽略；轻量
`combination_reports/` 可按根目录策略跟踪。`tests/glm5_2_combination/experiments/` 保存
人工整理的 Markdown/HTML/JSON 索引。原始日志、checkpoint、Profiler 数据不复制进本目录。

编译中间文件统一在仓库外：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/
```

## 证据规则

1. smoke 的 topology `manifest.json` 与 wrapper `report.md` 是能否启动和完成训练的依据。
2. precision 的 `precision_metrics.jsonl`、endpoint manifest 和 compare 报告是数值结论依据。
3. performance 的 `raw_metrics.jsonl` 是吞吐结论依据；本轮 profiler-off 比较排除 steps
   1-10，以 steps 11-30 为 steady-state，并对每个模式执行两个 repeat。
4. `--compare --require-all` 成功之前，不把全拓扑精度写成通过。
5. NPUGraphs 的 `PASSED_AOT_COMPAT` 只表示 replay 关闭后的 AOT 兼容路径，不等价于
   原生 NPUGraph capture/replay。
6. 本目录的 Markdown 是证据索引；原始运行目录不进入 Git，轻量正式 report 可跟踪。

建议阅读顺序：先看 [总结](reports/summary.md)，再看
[smoke 验证](reports/smoke-graph-validation.md)、[5000-step 精度](reports/precision-5000.md)、
[eager/Inductor 性能矩阵](reports/performance/summary.md)、
[失败历史](reports/failures.md) 和 [命令账本](history/commands.md)。
