# GLM-5.2 graph-mode debug

本目录只承载 NPU Inductor 图模式的软件栈调试入口、运行说明、环境适配报告和后续
bug 调试文档，不承载正式 precision、performance、stability 或 combination 验收。

## 内容

- `run_npu_inductor.sh`：隔离 CANN 9.1、复用 `.cache`、运行并自动生成报告。
- `RUN_NPU_INDUCTOR.md`：smoke/train/probe/env 用法、缓存和故障排查。
- `NPU_INDUCTOR_ENV_ADAPTATION_REPORT.md`：首轮环境兼容问题的完整证据与结论。
- `ARTIFACT_LAYOUT.md`：运行结果和 `.cache` 编译中间产物的目录规范。

## 目录约定

```text
tests/glm5_2_graph_debug/        # 仅脚本和文档，纳入 Git
graph_debug_runs/                # 调试运行结果、报告和日志，Git 忽略
~/.cache/torchtitan-test/graph_mode/
                                # Inductor/Triton/PCH/compile debug 中间产物
```

容器内 `~/.cache` 的持久化位置使用
`/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/`，对应宿主机
`/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/`。

快速运行：

```bash
tests/glm5_2_graph_debug/run_npu_inductor.sh env
tests/glm5_2_graph_debug/run_npu_inductor.sh smoke
```
