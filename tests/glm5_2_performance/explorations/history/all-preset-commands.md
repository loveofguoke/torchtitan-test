# 全拓扑 all-preset 性能实验命令账本

本文记录 2026-08-31 启动的完整 Ascend 性能采集。原始 profile、训练日志和
`msprof-analyze` 产物位于 Git 忽略目录；每个成功 run 的轻量 manifest、指标、精确命令
和输出索引由 performance workflow 自动同步到 `explorations/runs/`。

## 1. 实验范围

性能 runner 的 `--topology all` 包含 27 个实验拓扑：

- 1 卡：`single`；
- 2 卡：`ddp2`、`fsdp2`、`tp2`、`cp2`、`pp2`、`ep2`；
- 4 卡：`ddp4`、`fsdp4`、`tp4`、`cp4`、`pp4`、`ep4`、`fsdp2-tp2`；
- 8 卡：`ddp8`、`fsdp8`、`tp8`、`cp8`、`pp8`、`ep8`、`fsdp2-tp4`、
  `fsdp2-cp4`、`tp2-cp4`、`fsdp4-tp2`、`fsdp2-pp4`、
  `fsdp2-tp2-pp2`、`fsdp2-tp4-ep8`。

`--preset all` 展开为 8 个非冗余、相互独立的采集策略：`overview`、
`distributed`、`kernel`、`operator`、`memory`、`flamegraph`、`runtime` 和
`system`。因此本轮计划总量是 **27 × 8 = 216 次独立 capture**。`comparison` 不在
`all` 中；`standard` 的 Level 1 信息已被更深入的策略覆盖。

本轮统一训练口径是 30 steps、skip 10、warmup 2、active 5、local/global batch
8/64、sequence length 128、seed 61。实验使用 `--replicate 31` 形成新身份，不覆盖此前
探索结果。

## 2. 环境预检

在容器 `glm5-npu-dev` 的
`/workspace/y50064852_yyb/torchtitan-test` 执行：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env

source /usr/local/Ascend/cann-9.1.0/set_env.sh
export PATH=/root/miniconda3/envs/torchtitan-0803-graph-adapt/bin:$PATH
command -v msprof
/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli --help
python tests/glm5_2_performance/profiler_benchmark.py --help
```

核查结果：CANN 9.1.0、torch `2.14.0.dev20260805+cpu`、torch_npu `2.14.0`、
triton_ascend `3.2.1`、官方 `msprof` 和隔离安装的 `msprof-analyze` CLI 均可用。
`msprof-analyze` 不依赖 graph Conda 的 PATH，而由
`TORCHTITAN_MSPROF_ANALYZE` 显式指定，防止环境切换后静默使用错误解释器。

环境报告：

```text
graph_debug_runs/launcher-env-inductor-20260831-092219-2666177/
```

## 3. 资源隔离和队列

启动时 8 张卡正在执行已有的 5000-step eager 精度任务，资源所有者为 PID 2459052。
本轮没有 kill、暂停、修改或并发分享 NPU；独立 tmux 会话
`all_preset_0831` 每 60 秒检查该 PID，正常退出后再冷却 120 秒。

此前 `graph_precision_5000_0831` 会话还未使用 NPU，只处于等待状态。为避免两个 waiter
在资源释放后竞态启动，它被替换为统一队列；统一队列在性能矩阵完成后继续执行原来完全
相同的 deterministic Inductor candidate/compare，不改变精度实验内容。

队列文件和完整 stdout/stderr：

```text
graph_debug_runs/all-preset-20260831/queue.sh
graph_debug_runs/all-preset-20260831/queue.log
```

观察命令：

```bash
docker exec glm5-npu-dev /usr/bin/tmux capture-pane \
  -pt all_preset_0831:0 -S -200

docker exec glm5-npu-dev tail -n 200 \
  /workspace/y50064852_yyb/torchtitan-test/graph_debug_runs/all-preset-20260831/queue.log
```

## 4. 有界探针

资源释放后先运行完整矩阵的第一个成员：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli
export TORCHTITAN_MSPROF_ANALYZE_WORKERS=2

tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- \
  /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python \
  tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology single --preset overview \
  --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 \
  --analysis-tools all
```

该调用同时验证训练、torch_npu profiler、解析、advisor、HTML 报告和实验归档。后续完整
命令按相同 identity 识别该 run，不覆盖也不重复 capture。

## 5. 完整矩阵

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- \
  /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python \
  tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology all --preset all \
  --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 \
  --analysis-tools all
```

`--analysis-tools all` 的实际行为是：

1. 对 offline preset 调用 `torch_npu.profiler.profiler.analyse`；
2. 对每个 profile 调用 `msprof-analyze advisor all`；
3. 对捕获多个 rank 的 preset 调用 `msprof-analyze cluster -m all`；
4. 继续生成 `cluster_time_summary`、`free_analysis`；
5. 为每个实际 rank 生成 `communication_bottleneck`；
6. 生成包含 Timeline、通信、内存、火焰图、MindStudio handoff 和工具状态的 HTML。

只有一个 rank 被采集的策略会明确跳过 cluster；这是工具输入约束，不是实验缺失。

## 6. 输出组织

每个 run 的原始数据和轻量证据按卡数、拓扑和完整实验身份分层：

```text
performance_runs/<card-scope>/<topology>/<run-id>/
performance_artifacts/<card-scope>/<topology>/<run-id>/
performance_reports/<card-scope>/<topology>/<run-id>.html
tests/glm5_2_performance/explorations/runs/<card-scope>/<topology>/<run-id>/
```

每个 `explorations/runs/.../<run-id>/readme.md` 会列出：driver 命令、实际 torchrun、执行
顺序、训练/解析输出、artifact、HTML、source commit 和失败状态。全矩阵完成后，综合结论
写入 `explorations/reports/full-matrix/`；不得仅凭 queue 退出码宣称 216/216 完成。

## 7. 后续 deterministic 图精度

性能采集完全释放 NPU 后，统一队列恢复：

```bash
tests/glm5_2_combination/run_graph_precision_5000.sh inductor candidate
tests/glm5_2_combination/run_graph_precision_5000.sh inductor compare
```

它复用已完成的 fixture 和 eager reference，保持 5000 steps、两个 repeat、固定 token
plan 和 deterministic algorithms。没有使用 `--performance-nondeterministic`，最终只有
`compare --require-all` 成功后才能声明图模式精度矩阵通过。

## 8. 当前状态

截至 2026-08-31 09:25 CST：队列为 `WAITING_FOR_RESOURCE_OWNER`；216 次 capture 尚未
开始。该状态是对共享资源的主动隔离，不是 profiler 或图模式失败。后续以 queue log、
run manifests 和 suite report 更新本节。

## 9. 首跑失败与恢复

资源于 09:50 CST 释放后，队列清理了原任务遗留在 NPU2 的孤儿 rank，并在 120 秒冷却后
启动 `single/overview`。该 run 的 30-step 训练、torch_npu profiler capture 和 CANN 同步
解析均成功；CANN 解析耗时 2 分 21 秒。随后 advisor 失败：图环境入口通过 `env -i`
重建环境时没有透传 `TORCHTITAN_MSPROF_ANALYZE` 和
`TORCHTITAN_MSPROF_ANALYZE_WORKERS`，因此 workflow 无法发现隔离 CLI。队列按 fail-fast
约定在 1/216 停止，没有继续生成不完整报告。

修复仅扩展 `graph_env_common.sh` 的受控透传白名单，不改变训练、图编译或 profiler 配置。
验证命令：

```bash
bash -n tests/glm5_2_graph_debug/graph_env_common.sh
git diff --check

docker exec \
  -e TORCHTITAN_MSPROF_ANALYZE=/workspace/y50064852_yyb/.codex-tools/msprof-analyze-cli \
  -e TORCHTITAN_MSPROF_ANALYZE_WORKERS=2 \
  -w /workspace/y50064852_yyb/torchtitan-test glm5-npu-dev \
  /bin/bash -lc \
  'tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- /usr/bin/env'

docker exec -e ASCEND_RT_VISIBLE_DEVICES= \
  -w /workspace/y50064852_yyb/torchtitan-test glm5-npu-dev \
  /root/miniconda3/envs/torchtitan-0803/bin/python -m pytest -q \
  tests/unit_tests/test_glm5_2_performance.py \
  tests/unit_tests/test_glm5_2_graph.py
```

结果为 launcher `PASSED`、两个变量均存在、48 个单测通过。09:59 CST 使用原队列脚本
恢复；完整 capture manifest 使 `single/overview` 不重跑训练，只重新执行缺失的 advisor、
HTML 和归档阶段。另一个冗余 precision waiter 已在启动训练前停止，避免把首轮 fail-fast
误判为性能矩阵完成而与恢复队列竞态；deterministic precision 仍由主队列在性能矩阵后执行。

## 10. 多 rank 同步解析超时与恢复

恢复后的 `single`、`ddp2`、`ddp8`、`fsdp2`、`fsdp8` overview 完成，随后原始
`tp8/overview` 在 active window 后失败。rank0 正在同步导出 profiler 数据，其余 rank 已
继续进入 collective；TorchTitan 在 step 1 把训练 ProcessGroup timeout 调整为 100 秒，
rank0 解析超过该阈值后 rank2 首先 SIGABRT，torchrun 再终止其余 rank。该时间关系说明
根因是 rank-local 同步解析造成的控制流不对齐，不是 TP8 前向/反向、显存或 HCCL 链路失败。

通用修复是：NPU `world_size > 1` 且 preset 原始 `parse_mode=sync` 时，在 workflow 中把
effective parse mode 改为 `offline`。原始 profiler Level、shape、stack、memory 和 active
window 均不变；run identity 显式增加 `offline`，训练结束并销毁 ProcessGroup 后才调用
`torch_npu.profiler.profiler.analyse`。单卡仍保留 preset 原始 parse mode，GPU 和显式 async/
offline 也不受影响。

定向复验：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- \
  /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python \
  tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology tp8 --preset overview \
  --visible-devices 0,1,2,3,4,5,6,7 --replicate 31 \
  --analysis-tools all
```

run 为 `npu-tp8-bf16-s30-l8-b64-seq128-seed61-overview-r31-offline-7d88d2fc`：
30/30 steps 完成，ProcessGroup 正常销毁；CANN 离线解析耗时 3 分 11 秒，确实超过原训练
timeout；advisor、综合 HTML 和探索归档全部成功。新增策略单测后相关测试为 49 passed。

## 11. operator 原生崩溃、单变量隔离与恢复

2026-08-31 22:39 CST，首个 `operator` capture
`npu-single-bf16-s30-l8-b64-seq128-seed61-operator-r31-52aa78a7` 在 step 12、即 profiler
active window 开始后触发原生 SIGSEGV。训练已稳定完成前 12 步，且尚未进入 CANN 解析；
因此问题边界是采集器激活，不是模型训练、显存、HCCL 或 msprof-analyze。

失败现场原样保留后，执行单变量隔离：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- \
  /root/miniconda3/envs/torchtitan-0803-graph-adapt/bin/python \
  tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology single --preset operator \
  --visible-devices 0,1,2,3,4,5,6,7 --replicate 32 \
  --no-record-op-args --analysis-tools all
```

隔离 run
`npu-single-bf16-s30-l8-b64-seq128-seed61-operator-r32-sync-67e6a143` 保留原 preset 的
Level1、`record_shapes`、`with_flops`、ArithmeticUtilization、`op_attr` 和原始数据，只关闭
`record_op_args`。结果为 30/30 steps、CANN 同步解析（3 分 18 秒）、advisor、HTML 和归档
全部通过，故该实验把崩溃项收敛到 `record_op_args`。默认 `operator` preset 已改为安全组合；
CLI 仍保留显式 `--record-op-args`，供其他版本栈或专门的兼容性复验使用。

## 12. memory/stack 原生崩溃与职责拆分

原始 `memory` preset 在 single、ddp2、ddp8、fsdp8 上均于 step 17（active window
结束、`on_trace_ready` 进入）触发 SIGSEGV。单变量实验结果如下：

| replicate | 保留能力 | 结果 |
|---:|---|---|
| r31 | memory + shape + stack + modules + timeline | SIGSEGV |
| r32 | memory + shape + stack + modules | SIGSEGV |
| r33 | memory + shape | 30/30 PASSED；官方提示 52 条 incomplete memory records |
| r34 | memory + shape + stack | SIGSEGV |
| r35 | memory + shape + modules + timeline | SIGSEGV |

因此当前 torch_npu 2.7.1/CANN 9.1 的稳定边界是 `profile_memory + record_shapes`；
问题不是 topology，也不是 timeline 导出函数单独造成，而是 memory collector 与 stack/module
元数据组合在 native stop/callback 路径上不兼容。默认 suite 据此拆分职责：`memory` 只负责
内存事件，`flamegraph` 单独负责 stack/module 并离线解析，`runtime` 不再重复二者，而保留
Level2、shape、L2、op attr、GC 和 Host CPU/内存。失败 run 全部保留，官方 memory timeline
在本版本栈明确标记为 unavailable，不能把 allocator 数值或普通曲线冒充为 timeline。

后续 `flamegraph` 进一步证明，即使 `profile_memory=false`，`with_stack` 单独开启仍在
step 17 SIGSEGV；只保留 `with_modules` 也在同一位置 SIGSEGV。因此当前栈不能安全采集
Python folded stack 或 module hierarchy。安全的 `flamegraph` capture 只保留 Level0 CANN
DB/call-path proxy 并离线解析；报告必须把 Python flame graph 标为 unavailable，不能把普通
operator timeline 或按名称聚合的柱状图伪装成火焰图。

## 13. system 扩展采集器的环境边界

原始 `system/single` 在 step 17 后输出
`sudo: /usr/bin/msprof_data_collection.sh: command not found`，随后 profiler stop 持续无进展。
该脚本是扩展 Host system/I/O/MSTX 采集路径的官方环境依赖，当前容器未安装；继续等待或在
27 个 topology 上重复均不能产生有效数据。安全 system preset 采用已单独验证的能力并集：
runtime 已通过的 Host CPU/MEM/GC，加 distributed 已通过的 all-rank interconnection；关闭
disk/network/osrt/numa、system I/O 和 MSTX。报告把缺失能力标为 environment-unavailable，
并保留挂起 run 的 runtime log，不能把 pre/post `npu-smi` 快照冒充为这些系统时间序列。
