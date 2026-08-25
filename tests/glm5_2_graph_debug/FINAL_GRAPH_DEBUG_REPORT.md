# GLM-5.2 NPU 图模式调试完整报告

## 1. 结论

截至 2026-08-25，本轮中断恢复后的图模式调试已经完成。

- Inductor：smoke 定义的 15/15 个单卡、多卡和组合拓扑均完成 10/10 steps，原生
  Inductor 编译执行通过。
- NPUGraphs：15/15 个拓扑均完成 10/10 steps；当前 CANN 9.1.0、torch 2.14、
  torch_npu 2.14 组合不支持 GLM-5.2 的 grouped-mm 捕获及 TP `DeviceMesh` 录制，
  因而默认 profile 保留 AOT graph 执行并显式禁用 NPUGraph replay。这是“兼容降级
  通过”，不是原生 replay 通过。
- PyTorch 没有更换：仍为 `2.14.0.dev20260805+cpu`。
- 原有 `torchtitan-0803`、系统 CANN 9.0 和其他实验结果未被修改。适配使用复制出的
  Conda 环境、独立 CANN 目录、仓库外 `.cache` 和按进程 `env -i`。
- 没有执行 `git add`、`git commit` 或 `git push`。

最终汇总报告：

```text
graph_debug_runs/smoke-suite-inductor-20260825-172213-4113992/reports/report.md
graph_debug_runs/smoke-suite-npugraphs-20260825-172207-4113845/reports/report.md
```

逐问题失败历史、实际命令账本和按卡数/拓扑证据索引见
`experiments/index.md`。该目录是可提交的实验档案；原始日志、manifest 和 trainer
输出仍保留在 Git 忽略的运行目录中。

## 2. 软件与目录

| 项目 | 实际值 |
|---|---|
| 容器 | `glm5-npu-dev` |
| CANN | `/usr/local/Ascend/cann-9.1.0` |
| Conda | `/root/miniconda3/envs/torchtitan-0803-graph-adapt` |
| torch | `2.14.0.dev20260805+cpu` |
| torch_npu | `2.14.0` |
| triton_ascend | `3.2.1` |
| 调试源码/文档 | `tests/glm5_2_graph_debug/` |
| 调试调用报告 | `graph_debug_runs/` |
| smoke 结果 | `smoke_runs/` |
| 编译缓存 | `/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/` |

`tests/glm5_2_graph_debug/` 只保存脚本和文档。日志、manifest、trainer 输出和失败现场
不进入该目录；Inductor、Triton、PCH 与 compile debug 中间产物只进入仓库外 `.cache`。

## 3. 推荐入口

```bash
# 查看隔离后的完整环境
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env

# 原始训练参数可以直接透传，缺失的 compile 参数由入口补齐
tests/glm5_2_graph_debug/run_graph_mode.sh inductor train \
  --training.steps=10

# 单个 smoke 拓扑
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology fsdp2-tp4-ep8

# 全部 smoke 拓扑
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke

# 保留原 graph benchmark / combination 参数
tests/glm5_2_graph_debug/run_graph_mode.sh inductor performance \
  --capture candidate --topology fsdp8
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate --topology fsdp8
```

直接在容器默认 shell 运行会混入 CANN 9.0 路径，不属于本次验证环境。

## 4. common 环境的关键设置

common 入口先用 `env -i` 创建干净子进程，再在 Python 导入前加载：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
```

两种 profile 共享：独立 Conda、ATB、动态生成的 HCCL host 端口、600 秒 HCCL
建链超时、2400 秒进程组初始化超时、仓库外缓存、`aten.complex` DTensor strategy、
普通 PP metadata P2P、空 expert grouped-mm 保护以及零长度 Triton grid 保护。

Inductor 默认：

```bash
TASK_QUEUE_ENABLE=0
TORCHINDUCTOR_COMPILE_THREADS=1
NPU_INDUCTOR_FALLBACK_LIST=aten.sum,_c10d_functional.all_reduce
```

NPUGraphs 兼容 profile 默认：

```bash
TASK_QUEUE_ENABLE=0
TORCHTITAN_NPUGRAPH_SKIP_ALL=1
TORCHTITAN_NPUGRAPH_UNSAFE_OPS=aten._grouped_mm.default,\
torchtitan_graph_debug.safe_grouped_mm.default
```

若要继续验证原生 NPUGraph capture，可显式设置：

```bash
GRAPH_NPUGRAPH_SKIP_ALL=0 \
GRAPH_TASK_QUEUE_ENABLE=1 \
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke \
  --topology single
```

该命令用于复现和继续后端调试；在当前版本上不属于通过配置。

## 5. 实测矩阵

| 拓扑 | Inductor | NPUGraphs profile |
|---|---|---|
| `single` | PASSED | PASSED（AOT 降级） |
| `ddp2` | PASSED | PASSED（AOT 降级） |
| `ddp8` | PASSED | PASSED（AOT 降级） |
| `fsdp8` | PASSED | PASSED（AOT 降级） |
| `tp8` | PASSED | PASSED（AOT 降级） |
| `cp8` | PASSED | PASSED（AOT 降级） |
| `pp8` | PASSED | PASSED（AOT 降级） |
| `ep8` | PASSED | PASSED（AOT 降级） |
| `fsdp2-tp4` | PASSED | PASSED（AOT 降级） |
| `fsdp2-cp4` | PASSED | PASSED（AOT 降级） |
| `tp2-cp4` | PASSED | PASSED（AOT 降级） |
| `fsdp4-tp2` | PASSED | PASSED（AOT 降级） |
| `fsdp2-pp4` | PASSED | PASSED（AOT 降级） |
| `fsdp2-tp2-pp2` | PASSED | PASSED（AOT 降级） |
| `fsdp2-tp4-ep8` | PASSED | PASSED（AOT 降级） |

每个正式结果均有：

```text
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-<backend>-model/
  <topology>/manifest.json
  <topology>/runtime.log
  <topology>/trainer_output/
```

失败、数值无效和中断现场以时间戳后缀保留，正式判断只读取无后缀拓扑目录中的 passed
manifest。

## 6. 本轮问题、定位与修复

| 问题 | 证据/根因 | 处理 |
|---|---|---|
| 默认 shell 混入 CANN 9.0 | Python 导入前动态库路径错误 | `env -i` 后 source CANN 9.1 |
| Inductor `aten.sum` 无有效 Triton tile | UB 需求超过设备上限 | 最小 ACLNN fallback |
| 编译 collectives 数值不一致 | 编译后的 all-reduce 路径不稳定 | `_c10d_functional.all_reduce` fallback |
| TP `aten.complex` 无 DTensor strategy | torch 2.14 nightly 未注册 | launcher-only strategy |
| PP metadata batched P2P 不兼容 | HCCL metadata 通信异常 | 使用普通 P2P |
| EP 空 expert grouped-mm | CANN 对零分组/零行启动失败 | 补齐空组并提供显式 backward |
| 补齐 offsets 变为 int64 | `torch.cumsum` 默认整数提升 | `dtype=torch.int32` |
| 动态 Triton `coreDim=0` | guard 只匹配 `_numel`，漏掉 `xnumel` | 改为匹配所有 `*numel` 非 reduction 参数 |
| NPUGraph grouped-mm capture 失败 | CANN 返回 107030/107027 | 默认禁用 replay，保留 AOT graph |
| TP NPUGraph 录制失败 | graph tree 拒绝 21 个 `DeviceMesh(tp=8)` 输入 | 默认禁用 replay |
| TP+PP 第 6 步 HCCL timeout | 降级 profile 仍使用异步 task queue 1 | profile 默认改为 queue 0 |
| HCCL host 60000 端口冲突 | 共享宿主上其他容器占用 | 每次调用动态分配 base port |

最复杂 Inductor 拓扑的最终报告为：

```text
graph_debug_runs/smoke-suite-inductor-20260825-164804-4064806/reports/report.md
```

它完整运行到 step 10，最终 loss `3.82438`。最复杂 NPUGraphs 兼容拓扑最终 loss
`3.82464`，报告为：

```text
graph_debug_runs/smoke-suite-npugraphs-20260825-172013-4111556/reports/report.md
```

## 7. checkpoint 残余进程处理

恢复任务前，按用户授权只匹配以下 checkpoint 命令：

```text
tests.glm5_2_checkpoint.capture_resume
tests/glm5_2_checkpoint/checkpoint_benchmark.py
```

共发现 64 个残余进程。先发送 `TERM`，等待 10 秒后 64 个进程全部退出；没有使用
`KILL`，没有删除 checkpoint 文件。随后确认 8 张 NPU 无运行进程，再开始图模式测试。
未触碰其他用户进程。

## 8. 为什么 smoke 比单次 train 慢

“某个 `train.sh` 跑通”只覆盖一个参数组合；无 `--topology` 的 smoke 是 15 个拓扑
顺序执行。每个拓扑还必须完成 10 步，因为 EP 的空 token 问题直到第 8 步才暴露。

主要耗时包括：

1. 每个 rank 独立进行 Dynamo/AOT/Inductor lowering、Triton/CANN 编译和装载。
2. 8 卡拓扑首次编译会产生 8 份进程开销；为防主机内存和编译器竞争，每 rank 只开
   1 个编译 worker。
3. TP/CP/PP/EP 改变 graph、shape 和 collective，不能完全复用另一个拓扑的缓存。
4. PP 使用多个 microbatch；组合拓扑每个 step 包含更多 P2P、all-gather 和同步。
5. 首个 step 包含编译和缓存填充；后续 step 才代表稳定运行速度。
6. graph benchmark/combination 还会执行 reference、candidate 或重复轮次，不能与一次
   smoke train 的耗时直接比较。

缓存会降低同一 graph/shape 的复跑编译成本，但不能把 15 个不同拓扑合并成一次编译。

## 9. CANN 9.1.0 安装

完整的官方 run 包安装、已有容器复制方法、SHA-256、目录校验、按进程激活和卸载说明
见 `CANN_9_1_INSTALLATION.md`。本次使用独立目录，不修改 `/usr/local/Ascend/latest`
或现有 CANN 9.0 软链；容器重建后需要重新安装/复制 CANN 9.1.0。

## 10. 最终校验

交付前重新执行了以下检查：

- 逐项读取两个正式 smoke 根目录下的 30 个 `manifest.json`：Inductor 15/15、
  NPUGraphs profile 15/15 均为 `status: passed` 且 `return_code: 0`。
- 对 `graph_env_common.sh`、`run_graph_mode.sh`、`run_npu_inductor.sh` 和
  `run_smoke_graph.sh` 执行 `bash -n`：通过。
- `git diff --check`：通过。
- 检查 `tests/glm5_2_graph_debug/`：没有 `.log`、`.json` 或 `.pyc` 中间产物。
- 通过 common 的 `command` target 执行 graph 与 smoke 集成单测：17/17 通过。报告：

```text
graph_debug_runs/launcher-command-inductor-20260825-175353-4115183/reports/report.md
```

其中此前单独执行的 graph 单测也为 13/13 通过，报告保留在
`graph_debug_runs/launcher-command-inductor-20260825-172749-4114584/`。

- `train_npu.py` 的 Python bytecode 编译检查通过。报告：

```text
graph_debug_runs/launcher-command-inductor-20260825-172514-4114273/reports/report.md
```

- 最终 Inductor 与 NPUGraphs profile 环境快照均通过：

```text
graph_debug_runs/launcher-env-inductor-20260825-172537-4114462/reports/report.md
graph_debug_runs/launcher-env-npugraphs-20260825-172537-4114523/reports/report.md
```

为完整保留调试历史：曾有一次把 shell 脚本路径误传给 `python -m py_compile`，因此该
诊断调用按预期失败；失败报告保留在
`graph_debug_runs/launcher-command-inductor-20260825-172508-4114206/`。随后已分别用
`bash -n` 校验 shell、用 `py_compile` 校验 `train_npu.py`，两项均通过。该误调用不属于
图模式运行失败。

## 11. 验收边界

本报告证明当前隔离软件栈、入口、15 个 smoke 拓扑和兼容修复能够稳定运行。它不是
正式 precision/performance/stability 验收；正式结论仍由原有 graph benchmark 和
combination 实验产生。尤其 NPUGraphs 当前只证明 AOT graph 降级执行可用，不能作为
NPUGraph replay 性能或正确性结论。
