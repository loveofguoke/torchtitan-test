# 图模式 common 与 smoke 验证报告

> 本文保留 common 层实现阶段和中断时的验证快照。中断恢复后的最终结果为
> Inductor 15/15 通过、NPUGraphs profile 15/15 AOT 兼容降级通过；请以
> `FINAL_GRAPH_DEBUG_REPORT.md` 和两个最终汇总报告为准。

## 1. 结论

截至 2026-08-25，图模式已统一为同一个 CANN 9.1.0/Conda/HCCL/cache common
环境。正常训练、现有 smoke、graph precision/performance/probe、combination 和任意
仓库命令均可保留原参数，通过 `run_graph_mode.sh` 选择 `inductor` 或 `npugraphs`。

当时实测结论（历史中断快照）：

- common 环境解析、CANN 9.1 激活、benchmark/combination 参数透传和 zero-numel
  两条 launcher dispatch 探针通过；
- Inductor smoke 15 种标准拓扑中 14 种完成 10/10 steps；
- 剩余 `fsdp2-tp4-ep8` 的最新代码修复已通过无设备 dispatch 探针，但 8 卡最终复验
  被共享设备的 stream 资源耗尽阻塞，不能标为 PASSED；
- NPUGraphs 当时只实跑通过 `single`、`ddp2`，均包含 `_grouped_mm` 子图的安全捕获
  降级；当时其余 13 种拓扑尚未实跑。最终结果以页首链接文档为准。

## 2. 软件栈与隔离边界

| 项目 | 实际值 |
|---|---|
| 容器 | `glm5-npu-dev` |
| NPU | Ascend 910B2 × 8 |
| Driver | `26.0.rc1` |
| CANN | `/usr/local/Ascend/cann-9.1.0` |
| Conda | `/root/miniconda3/envs/torchtitan-0803-graph-adapt` |
| torch | `2.14.0.dev20260805+cpu` |
| torch-npu | `2.14.0` |
| triton-ascend | `3.2.1` |
| ATB | `/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1` |
| 编译 cache | `/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321` |
| 调试运行 | `graph_debug_runs/` |
| smoke 业务结果 | `smoke_runs/` |

common 使用 `env -i` 后重新执行入口，再 source CANN 9.1.0；不会修改 CANN 9.0
全局软链，也不会修改原 `torchtitan-0803` 环境。编译中间结果全部保存在宿主机主目录
挂载的 `.cache`，没有进入 Git 仓库。

## 3. 统一入口验证

### 3.1 环境 profile

命令：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs env
```

结果：PASSED。确认 CANN、Conda、ATB、HCCL、task queue、NPUGraph unsafe ops、
两项动态空输入保护及三个编译 cache 均来自 common。关键值为：

```text
ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.1.0
TASK_QUEUE_ENABLE=1
TORCHTITAN_NPUGRAPH_UNSAFE_OPS=aten._grouped_mm.default,torchtitanturbo_graph.safe_grouped_mm.default
TORCHTITAN_SAFE_EMPTY_GROUPED_MM=1
TORCHTITAN_SAFE_ZERO_NUMEL_TRITON=1
```

2026-08-25 11:31 重新执行后的两份环境报告：

```text
graph_debug_runs/launcher-env-inductor-20260825-113129-3817988/reports/report.md
graph_debug_runs/launcher-env-npugraphs-20260825-113130-3818044/reports/report.md
```

### 3.2 graph benchmark 原参数透传

命令：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor performance --list-topologies
```

结果：PASSED，原入口列出全部 15 种拓扑，wrapper 自动补充 graph pair，没有改写
原参数。调用报告：

```text
graph_debug_runs/launcher-performance-inductor-20260825-113130-3818101/reports/report.md
```

### 3.3 combination 原参数透传

命令：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination --list-topologies
```

结果：PASSED，原入口列出全部 15 种拓扑。调用报告：

```text
graph_debug_runs/launcher-combination-inductor-20260825-113130-3818163/reports/report.md
```

### 3.4 zero-numel dispatch 保护

不初始化 NPU，分别直接调用普通和 symbolic-grouped autotuner，令
`x0_numel=0`。两条路径均打印 skip，并在 original launch 前返回。结果：PASSED。

```text
[torchtitan-npu] skipped zero-numel Triton kernel: zero_numel_guard_probe (x0_numel=0)
[torchtitan-npu] skipped zero-numel Triton kernel: zero_numel_guard_probe (x0_numel=0)
graph_debug_runs/launcher-command-inductor-20260825-091227-3808822/reports/report.md
```

### 3.5 静态与原有单元测试

以下检查全部通过：

```text
bash -n graph_env_common.sh run_graph_mode.sh run_smoke_graph.sh
python3 -m py_compile train_npu.py
git diff --check
pytest test_glm5_2_smoke.py test_glm5_2_graph.py: 17 passed
```

pytest 通过 common `command` target 执行，报告为：

```text
graph_debug_runs/launcher-command-inductor-20260825-113130-3818225/reports/report.md  # bash -n
graph_debug_runs/launcher-command-inductor-20260825-113130-3818287/reports/report.md  # py_compile
graph_debug_runs/launcher-command-inductor-20260825-113131-3816828/reports/report.md  # 17 pytest
```

## 4. Inductor smoke 矩阵

统一 contract 为 10 steps、global batch 64、sequence length 128、seed 61、
`glm5_debugmodel`。

| 拓扑 | 结果 |
|---|---|
| `single` | PASSED |
| `ddp2` | PASSED |
| `ddp8` | PASSED |
| `fsdp8` | PASSED |
| `tp8` | PASSED |
| `cp8` | PASSED |
| `pp8` | PASSED |
| `ep8` | PASSED |
| `fsdp2-tp4` | PASSED |
| `fsdp2-cp4` | PASSED |
| `tp2-cp4` | PASSED |
| `fsdp4-tp2` | PASSED |
| `fsdp2-pp4` | PASSED |
| `fsdp2-tp2-pp2` | PASSED |
| `fsdp2-tp4-ep8` | 待最新修复的 8 卡复验 |

14 个 PASSED 结论均有当前目录下的 manifest：

```text
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-inductor-model/<topology>/manifest.json
```

`fsdp2-tp4-ep8` 的后端失败链为：rank 7 在 step 8 接收 0 个远端 token；动态
pointwise 首先用零 grid 启动 CANN kernel，queue=0 时为 `EE1003 coreDim=0`，异步
表现可能为 SIGSEGV。进一步还需要保护空 expert 的 grouped-mm。当前实现分别：

1. 在普通与 symbolic-grouped Triton autotuner launch 前跳过非 reduction
   `*_numel=0` 的空 pointwise；
2. 对部分空 expert 临时补零行，执行后抽取原行；总 token 数为 0 时完全绕过
   grouped-mm；
3. 反向按原 offsets 显式计算，不让补行产生参数梯度。

旧代码的最后一份确定性后端失败报告：

```text
graph_debug_runs/smoke-suite-inductor-20260824-211716-3560240/reports/report.md
```

最新代码复验在第一步 FSDP 创建 stream 时被基础设施阻塞，尚未进入图编译：

```text
Resource_Error(EE1023): Alloc Stream resource failed. Reason: Too many streams are created.
graph_debug_runs/smoke-suite-inductor-20260825-090754-3805779/reports/report.md
```

当时 `npu-smi info` 显示每张卡约有 8 个其他进程、HBM 占用约 34–39 GiB，且另一个
PP8 性能实验正在运行。没有结束或修改这些外部进程。

2026-08-25 11:31 后继续只读跟踪，确认当前占卡任务是：

```text
tests/glm5_2_checkpoint/checkpoint_benchmark.py
  --failure-mode all --topology=fsdp4-tp2
```

该套件在 `sigkill` 场景杀掉 torchrun 后，原 8 个 rank 进程仍可见；恢复阶段又启动
新的 8-rank torchrun。它与此前每卡累积多个 Python 进程及 `EE1023 Too many
streams` 的现场一致。没有向 checkpoint 调度器、旧 rank 或恢复进程发送信号；在
这些进程释放前，不应继续启动 8 卡图模式测试。

再次恢复任务时的现场复查仍未释放：总调度器 PID `3815593` 正在运行
`checkpoint_benchmark.py --topology all --failure-mode all`，当前推进到后续组合拓扑；
`npu-smi info` 显示每张卡仍有 7 个 Python 进程、约 22 GiB HBM，并可同时看到旧
`sigkill`/`incomplete` rank 与新的恢复或 `rank-error` rank。该状态仍有明确的 stream
和 HCCL 竞争风险，因此没有启动新的图模式训练，也没有自行终止 checkpoint 进程。

## 5. NPUGraphs smoke 矩阵

| 拓扑 | 结果 | 报告 |
|---|---|---|
| `single` | PASSED（含安全降级） | `graph_debug_runs/smoke-suite-npugraphs-20260824-181603-1824452/reports/report.md` |
| `ddp2` | PASSED（含安全降级） | `graph_debug_runs/smoke-suite-npugraphs-20260824-192421-760050/reports/report.md` |
| 其余 13 种 | 未实跑 | 设备资源阻塞后未继续占用共享卡 |

“含安全降级”表示 AOT graph 与训练完成，但含 `_grouped_mm` 的 FX 子图因 CANN 9.1
捕获期间同步 device offset 不安全而跳过 NPUGraph replay。它不是完整 replay 通过。

## 6. 已适配问题

| 问题 | 处理 |
|---|---|
| 默认 shell 泄漏 CANN 9.0 | `env -i`，Python 前 source CANN 9.1 |
| 编译结果进入仓库或重复冷编译 | Inductor/Triton/debug 全部固定到主目录 `.cache` |
| NPU NIC 与 host HCCL 端口冲突 | `auto` NPU port + 每调用独立 host base port |
| 冷编译超过通信超时 | HCCL 600 秒、process group 2400 秒 |
| Turbo 把 task queue 重写为 2 | bootstrap 在 Turbo 后恢复 common 请求值 |
| TP `aten.complex` 缺 DTensor strategy | launcher-only 注册 broadcast-aware strategy |
| PP metadata batched P2P 损坏 | metadata 改用普通 P2P |
| `_grouped_mm` 无法安全 NPUGraph 捕获 | unsafe-op 扫描并安全跳过对应 replay 子图 |
| 空 expert grouped-mm 零核启动 | launcher-only padding/bypass + 显式 backward |
| 动态空 pointwise 零 grid 启动 | base/grouped autotuner launch 前 no-op |
| 8 rank 编译 worker 过多 | 每 rank 默认 1 个 worker |

## 7. 设备空闲后的续跑命令

先确认目标卡空闲：

```bash
npu-smi info
```

再只复验缺口；已有 passed manifest 会被 smoke runner 自动跳过：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology fsdp2-tp4-ep8
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

最终只有 exit code 0、10/10 steps、数值检查通过且产生 passed manifest，才更新为
PASSED。`EE1023`、507033 或其他设备初始化资源错误应记录为基础设施失败。

## 8. 本轮完整执行步骤

1. 盘点容器默认 CANN、原 `torchtitan-0803`、torch/torch_npu/triton 版本和现有
   smoke/graph/combination CLI，确认不修改原环境与原实验 contract。
2. 从 `torchtitan-0803` 复制独立 Conda 环境并在副本中适配；PyTorch 保持
   `2.14.0.dev20260805+cpu`，CANN 9.1.0 以独立目录安装且不切换全局软链。
3. 将 CANN 9.1、Conda、ATB、HCCL、Inductor/NPUGraphs 和仓库外 `.cache` 抽为
   `graph_env_common.sh`，再由 `run_graph_mode.sh` 统一包装 train、smoke、三个 graph
   benchmark、combination 与任意命令。
4. 用 single Inductor 定位 reduction UB overflow；保留正式 NPU codegen，只最小
   fallback `aten.sum`。之后逐步扩展 DDP/FSDP/TP/CP/PP/EP 和组合拓扑。
5. 分别定位并修复 TP `aten.complex` DTensor strategy、PP metadata P2P/HCCL
   timeout、Inductor collective 数值错误、EP 空 expert grouped-mm 和动态零 grid。
6. 用 single 与 DDP2 验证 NPUGraphs；确认 `_grouped_mm` 无法安全 capture 后采用
   FX 子图级安全降级，并在报告中明确它不是完整 replay。
7. 验证 benchmark 与 combination 原参数透传、环境 profile、zero-numel 两条
   autotuner dispatch、Python 语法、Shell 语法、diff whitespace 和原有单元测试。
8. 恢复最后一个 8 卡 Inductor 拓扑时，先读取设备现场；每卡约 8 个外部 checkpoint
   进程、HBM 约 32–39 GiB，实际在 FSDP 创建 stream 时返回 EE1023。没有结束、修改
   或抢占这些进程，失败报告按基础设施问题保留。
9. 继续等待时发现 checkpoint `sigkill` 场景保留了原 rank，同时启动恢复 torchrun，
   因此设备不会在普通场景切换间稳定释放。停止了本任务自己的只读监控循环，但未
   中止或清理任何 checkpoint 进程。待用户结束该套件并释放设备后，只续跑缺少
   passed manifest 的 `fsdp2-tp4-ep8` 和剩余 NPUGraphs 拓扑，不重做或删除已通过
   结果。

## 9. train 与 smoke 耗时差异

此前说 `run_train.sh` 已跑通，指特定 debug 配置和已选拓扑能够使用 Inductor；并不
代表 15 个 smoke 并行 contract 或 NPUGraphs 矩阵都已验证。smoke 的默认 `all` 是
15 个拓扑串行，每个拓扑内部最多 8 rank 各自冷编译 forward/backward，随后运行
10 steps 检查动态路径和数值稳定性。TP/CP/PP/EP 的 graph key 与 single 不同，缓存
只能部分复用；为避免 CANN 多 worker 资源错误，编译并发还被限制为每 rank 1。

这也是 smoke 看起来“训练很短、总时间很长”的原因：时间主要花在每个新拓扑的
首步编译、autotune、HCCL 建链和失败后的定向复验，而不是 10 个训练 step 本身。
single 正式运行约 172 秒，TP8 冷编译约 683 秒；已经通过的 manifest 再次调用约
1 秒即可跳过。EP 的零 token 错误直到 step 8 才触发，也证明不能用 1-step 编译成功
替代完整 smoke 结论。
