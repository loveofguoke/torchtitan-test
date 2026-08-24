# 图模式调试产物目录规范

## Git 内

```text
tests/glm5_2_graph_debug/
```

仅保存可维护的脚本和文档，不保存运行日志、报告实例、PCH、NPUBIN、Inductor
cache、Triton cache 或 `torch_compile_debug`。

## 调试运行结果

```text
graph_debug_runs/<action>-<timestamp>-<pid>/logs/runtime.log
graph_debug_runs/<action>-<timestamp>-<pid>/reports/report.md
```

历史环境适配结果归档在：

```text
graph_debug_runs/environment-adaptation-20260824/reports/
graph_debug_runs/environment-adaptation-20260824/logs/
```

`graph_debug_runs/` 与其他实验的 `*_runs/` 一样由 Git 忽略。

## 编译中间结果

宿主机：

```text
/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/
```

容器内：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/
```

目录内容：

```text
inductor/                 # TORCHINDUCTOR_CACHE_DIR
triton/                   # TRITON_CACHE_DIR
torch_compile_debug/      # TORCH_COMPILE_DEBUG_DIR
adaptation_artifacts/     # 首次适配保留的 PCH、失败缓存和 compile debug
```

这些目录不在 Git 工作树中，但通过宿主机到容器的主目录挂载保持持久并可复用。
