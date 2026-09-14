# 多卡 NPU Inductor 图模式调试

## 范围

本文件持续记录多卡图模式的环境、问题、修复和复测。先分别验证数据并行的两种
基本形态：

- `ddp2`：模型副本在两卡复制，梯度跨卡同步；
- `fsdp2`：模型参数在两卡分片，forward/backward 包含参数 all-gather 与梯度
  reduce-scatter。

本轮同时扩展到 `ddp4`/`fsdp4` 和仓库常用的 `ddp8`/`fsdp8`。后续再扩展到
TP/CP/PP/EP 组合。单卡通过不能替代多卡验证，因为多卡会
增加 HCCL collective、DTensor/FSDP wrapping、跨 rank 编译缓存并发和错误传播路径。

## 运行入口

默认使用空闲的物理 NPU 3、4：

```bash
tests/glm5_2_graph_debug/run_npu_inductor.sh smoke-ddp
tests/glm5_2_graph_debug/run_npu_inductor.sh smoke-fsdp
```

指定卡和规模：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 GRAPH_NGPU=2 \
  tests/glm5_2_graph_debug/run_npu_inductor.sh smoke-ddp
```

四卡示例：

```bash
ASCEND_RT_VISIBLE_DEVICES=3,4,5,6 GRAPH_NGPU=4 \
  tests/glm5_2_graph_debug/run_npu_inductor.sh smoke-fsdp
```

普通训练入口：

```bash
tests/glm5_2_graph_debug/run_npu_inductor.sh train-ddp
tests/glm5_2_graph_debug/run_npu_inductor.sh train-fsdp
```

## 产物

每次运行按原调试规范写入：

```text
graph_debug_runs/<action>-<timestamp>-<pid>/logs/runtime.log
graph_debug_runs/<action>-<timestamp>-<pid>/reports/report.md
```

所有 rank 继续共享主目录下的持久编译缓存：

```text
/workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/
```

报告额外记录 topology、world size 和物理 NPU 列表。

## 测试记录

| 编号 | 拓扑 | 状态 | 耗时 | 结果目录 |
|---|---|---:|---:|---|
| M01 | ddp2 | 资源冲突中止 | - | `smoke-ddp-20260824-154206-976772` |
| M02 | ddp2 | **PASS** | 224 s | `smoke-ddp-20260824-154626-1178208` |
| M03 | fsdp2 | **PASS** | 64 s | `smoke-fsdp-20260824-155016-1397416` |
| M04 | ddp4 | **FAIL（设备瞬态）** | 38 s | `smoke-ddp-20260824-155129-1406817` |
| M05 | ddp4 | **PASS** | 245 s | `smoke-ddp-20260824-155243-1407460` |
| M06 | fsdp4 | **PASS** | 99 s | `smoke-fsdp-20260824-155654-1424140` |
| M07 | ddp8 | **FAIL（HCCL 端口）** | 37 s | `smoke-ddp-20260824-160139-1440860` |
| M08 | ddp8 | **PASS** | 264 s | `smoke-ddp-20260824-160345-1442387` |
| M09 | fsdp8 | **PASS** | 105 s | `smoke-fsdp-20260824-160815-1478518` |

以上目录均位于 `graph_debug_runs/`，每个目录都有 `logs/runtime.log` 和
`reports/report.md`。所有通过项均完成 10/10 steps、出现一次
`Training completed`、退出码为 0，且报告中的 compiler/backend error 计数为 0。

## 问题与处理

### M01：外部全卡任务资源冲突

DDP2 已完成 HCCL 初始化、构建设备 mesh 并进入 Inductor 编译后，另一个实验启动
`ddp8` 并占用全部物理卡。为避免两个实验互相污染，主动停止本次 graph debug。
该次结果标记为 `ABORTED_RESOURCE_CONFLICT`，不计作图模式功能失败；停止后已确认
graph debug 的 torchrun、rank 和 compile worker 全部退出。

### M04：NPU 5 瞬态 `TsdOpen failed`

第一次 DDP4 在 rank 2 初始化物理 NPU 5 时失败，发生在模型构建和 Inductor 编译前：

```text
error code is 507033
Failed to start the device
TsdOpen failed. devId=5
```

当时前一个全卡任务刚退出。`npu-smi` 随后显示设备无进程且 Health 为 OK；在同一
CANN 9.1/Conda 隔离环境下，单独对物理 NPU 5 执行 `set_device` 和 NPU 张量计算
成功。未重置设备、未改环境，原命令重跑即通过 10/10 steps。因此该问题判定为设备
释放窗口中的瞬态运行时状态，而非 Inductor、DDP 或 CANN 版本配套缺陷。

处理原则：

1. 先确认目标卡无其他进程，并等待前一任务彻底退出；
2. 用目标卡进行最小 `set_device` + 张量计算；
3. 最小验证成功后原样复测；
4. 不在共享机器上擅自 reset NPU。

### M07：HCCL NPU 默认端口跨容器冲突

第一次 DDP8 在确定性 seed broadcast 创建通信域时失败，NPU 1、2 均报告：

```text
Communication_Error_Bind_IP_Port(EI0020)
IP address ... and port 16666 have already been bound
configure HCCL_NPU_SOCKET_PORT_RANGE
```

容器内没有其他 NPU 进程，但 NPU 网卡属于宿主资源，其他容器可以占用相同默认端口。
2/4 卡没有触发并不能证明 8 卡通信平面的所有 NPU IP 都没有端口冲突。

修复是在 `run_npu_inductor.sh` 的 `env -i` 子进程内设置：

```text
HCCL_NPU_SOCKET_PORT_RANGE=auto
```

华为 HCCL 环境变量说明支持具体端口、端口范围或 `auto`；未配置时默认使用 16666，
并要求同一通信域所有卡使用相同配置。脚本提供
`GRAPH_HCCL_NPU_SOCKET_PORT_RANGE` 覆盖项，便于管理员使用固定规划范围。该变量不会
写入当前 shell 或全局环境。

修复后 DDP8 和 FSDP8 都完成 10/10 steps，证明端口冲突已消除且 8 卡图模式可用。

## 编译耗时观察

- DDP2 首次完整运行 224 秒，DDP4 首次完整运行 245 秒；多 rank 并发会竞争 CPU
  编译 worker、缓存锁和文件系统。
- FSDP2 为 64 秒、FSDP4 为 99 秒，说明已生成的模型 Inductor/Triton 缓存可以跨
  DDP/FSDP 包装和后续进程复用。
- DDP8 为 264 秒，FSDP8 为 105 秒；8 rank 同时加载或补编译时竞争更明显。
- 2/4 卡通过项进入稳态后的单步约 0.18–0.21 秒；FSDP8 约 0.43–0.75 秒/步。
  主要耗时仍在 step 1 前。

## 当前结论与边界

CANN 9.1.0、torch 2.14 nightly、torch_npu 2.14.0、Triton-Ascend 3.2.1 组合下，
GLM-5.2 debugmodel 的 Inductor 图模式已在 DDP2、FSDP2、DDP4、FSDP4、DDP8、
FSDP8 跑通。物理 NPU 0 的 Health 虽持续显示 `Warning`，但本轮 DDP8/FSDP8 均能
完成；后续长时间或正式验收仍应单独跟踪该硬件告警。本目录的 smoke 结果不是正式
precision、performance 或 stability 验收。
