# NPU Inductor 图模式运行手册

本目录用于图模式环境调试和快速定位，不作为 precision、performance、stability
或 combination 的正式验收结果。`probe` 子命令复用既有实验框架时，仍遵循该框架
原有的结果目录与报告规范；`smoke/train` 的调试日志则统一写到用户主目录。

## 推荐入口

在 `glm5-npu-dev` 容器内运行：

```bash
cd /workspace/y50064852_yyb/torchtitan-test

# 查看实际加载的软件栈，不运行训练
tests/glm5_2_graph_debug/run_npu_inductor.sh env

# 推荐：10-step 快速图模式验证
tests/glm5_2_graph_debug/run_npu_inductor.sh smoke

# 原始普通训练入口
tests/glm5_2_graph_debug/run_npu_inductor.sh train

# 现有测试框架的 eager reference -> Inductor candidate -> compare 全流程
tests/glm5_2_graph_debug/run_npu_inductor.sh probe
```

脚本默认使用物理 NPU 2。使用其他卡：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 \
  tests/glm5_2_graph_debug/run_npu_inductor.sh smoke
```

`train` 和 `smoke` 后可继续传递 TorchTitan 参数：

```bash
tests/glm5_2_graph_debug/run_npu_inductor.sh train \
  --training.steps=100 \
  --training.max_context_length=128
```

## 三种运行模式

### smoke

直接复用仓库中 `compile_probe.py` 的快速规格：10 steps、batch 1、sequence
length 32。它覆盖 Dynamo、AOTAutograd、Inductor、Triton-Ascend、launcher、
forward、backward 和 optimizer，适合判断环境是否可用。

### train

调用仓库原有 `run_train.sh`，自动添加：

```text
--compile.enable
--compile.components=model
--compile.backend=inductor
```

固定 `NGPU=1 MODULE=glm5 CONFIG=glm5_debugmodel`，其他训练参数保持 TorchTitan
配置默认值或使用命令行覆盖。

### probe

直接编排已有 `tests/glm5_2_graph/compile_probe.py`：

1. 生成或复用固定 token/checkpoint 数据；
2. 采集 NPU eager reference；
3. 采集 NPU Inductor candidate；
4. 比较精度和编译诊断，并要求全部检查通过。

这是功能验证最完整的入口，但耗时比 `smoke` 更长。

## 为什么第一次慢

第一次运行需要完成整套冷编译：

```text
Dynamo capture
  -> AOTAutograd forward/backward graph
  -> Inductor lowering and scheduling
  -> Triton-Ascend code generation
  -> TTIR / adapter / NPUBIN
  -> launcher PCH/shared object compilation
  -> autotune candidate pre-run and benchmark
```

本次 debugmodel 冷编译实测约 7 分 35 秒，生成约 5,546 个 `.npubin` 候选，
编译与诊断目录约 977 MB。训练稳态本身只有约 0.18–0.20 秒/step，因此绝大部分
首轮时间是编译和 autotune，而不是模型执行。

## 如何缩短后续运行

脚本固定复用以下缓存：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/inductor
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/triton
```

同一代码、shape、dtype、版本和编译选项再次运行时会命中大量缓存。本次同缓存复跑
总时长为 80 秒，step 2–10 平均约 173.2 tokens/s。

缓存 key 会受代码、shape、dtype、torch/torch_npu、Triton-Ascend 和编译配置影响。
这些条件变化时重新编译属于正常现象。不要在已验证缓存中混用 CANN 9.0 与 9.1。

如需为不同实验隔离缓存：

```bash
GRAPH_CACHE_ROOT=/workspace/cache/graph-experiment-a \
  tests/glm5_2_graph_debug/run_npu_inductor.sh smoke
```

如需分别指定已有的两个缓存目录，可使用 `GRAPH_INDUCTOR_CACHE_DIR` 和
`GRAPH_TRITON_CACHE_DIR`。日常使用只设置 `GRAPH_CACHE_ROOT` 即可。

## 日志与报告

每次运行都会创建：

```text
graph_debug_runs/<action>-<timestamp>-<pid>/logs/runtime.log
graph_debug_runs/<action>-<timestamp>-<pid>/reports/report.md
```

`runtime.log` 包含实际训练命令、step、编译日志和异常栈；`report.md` 包含环境、
状态、退出码、耗时和缓存路径。现有 `compile_probe.py` 仍会另外生成它原有的
`combination_runs/`、`combination_artifacts/` 和精度比较产物。

`graph_debug_runs/` 与仓库其他 `*_runs/` 一致，属于生成结果并由 Git 忽略。
源码子目录只保留文档和脚本。大体积编译内容位于宿主机
`/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/`，容器重启后仍可复用。

## 环境隔离

脚本通过 `env -i` 启动子进程，只启用：

```text
/root/miniconda3/envs/torchtitan-0803-graph-adapt
/usr/local/Ascend/cann-9.1.0
/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1
```

它不会修改原 Conda 环境、当前 shell、Driver、`/usr/local/Ascend/cann` 或
`/usr/local/Ascend/ascend-toolkit/latest` 软链。

## 常见问题

- 出现 `aclmdlRICondHandle has not been declared`：进程混入了 CANN 9.0 include。
  先执行 `run_npu_inductor.sh env` 并检查 CANN 路径。
- 出现 `No valid triton configs`：它是上层包装错误，必须向前查找第一个 launcher、
  g++ 或 kernel 错误，不能只按 autotune 问题处理。
- 修改 shape 后再次变慢：新 shape 触发新图和新 kernel 编译，属于预期。
- 希望强制冷启动：为 `GRAPH_CACHE_ROOT` 指定一个新的空目录，不要删除共享缓存。
