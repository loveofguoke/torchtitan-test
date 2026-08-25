# GLM-5.2 NPU 图模式调试、跑通与三仓修复记录

## 1. 当前结论与验收边界

本轮服务器实验基于 Ascend 910B2 × 8、CANN 9.1.0、torch/torch_npu 2.14
nightly 和 triton-ascend 3.2.1。统一 smoke contract 为 10 steps、global batch
64、sequence length 128、seed 61、`glm5_debugmodel`。

- NPU Inductor：15/15 个标准拓扑完成 10/10 steps。它仍对 `aten.sum` 和
  `_c10d_functional.all_reduce` 使用最小 fallback，因此是“Inductor 主体编译 +
  已知不稳定算子回退”，不是零 fallback 全图通过。
- NPUGraphs：15/15 个拓扑完成 10/10 steps，但默认
  `TORCHTITAN_NPUGRAPH_SKIP_ALL=1`。Dynamo/AOT 图仍执行，NPUGraph replay 被显式
  禁用，因此只能称为 AOT 兼容降级通过，不能称为原生 NPUGraph 通过。
- 以上 15/15 结果来自兼容逻辑仍位于 test launcher 时的服务器实验。相同逻辑现已
  迁入 TorchTitanTurbo 的 opt-in 模块；三仓职责重构后的源码安装复验尚未完成，不能
  把“已实现”误写成“新版本已验证”。
- 本轮是跑通性 smoke，不替代 eager-vs-graph 的 5000-step precision、performance、
  checkpoint 或 stability 验收。

完整原始调试证据保留在 `tests/glm5_2_graph_debug/`；本文给出面向后续研发的统一
结论、命令、修复归属和剩余问题。

## 2. 从单卡到全部拓扑的跑通过程

### 2.1 环境隔离

默认容器 shell 混有 CANN 9.0，而 torch_npu 2.14 编译依赖 CANN 9.1 header/API。
`graph_env_common.sh` 使用 `env -i` 建立干净进程，在 Python 导入前 source CANN
9.1，并选择独立 Conda、ATB、HCCL 配置和仓库外编译缓存。只在 Python 内修改
`LD_LIBRARY_PATH` 无效，因为 torch_npu 已经加载旧动态库。

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs env
```

### 2.2 单卡定位

Inductor 单卡首先暴露 fused reduction 的 UB tile 超限。保留正式 NPU codegen，仅将
`aten.sum` 回退 ACLNN 后完成 10 steps。NPUGraph 单卡随后暴露 grouped-mm 捕获期间
device offset 同步失败；`thread_local` 与 `relaxed` 均不能解决，因此保留 AOT 并将
replay 标记为不兼容。

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single
ASCEND_RT_VISIBLE_DEVICES=0 \
  tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke --topology single
```

### 2.3 多卡逐层扩展

多卡按 DDP/FSDP、TP/CP、PP、EP、组合拓扑推进，分别定位：HCCL NIC/host 端口冲突、
TP `aten.complex` 缺 DTensor strategy、PP metadata batched P2P 损坏、冷编译建链
timeout、编译 all-reduce 数值异常、EP 空 expert grouped-mm 与动态零 grid。

单个分布式拓扑：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology fsdp2-tp4
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke --topology fsdp2-tp4
```

指定集合：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topologies ddp8,fsdp8,tp8,pp8,ep8
```

完整矩阵：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

`all` 覆盖 single、ddp2、ddp8、fsdp8、tp8、cp8、pp8、ep8、fsdp2-tp4、
fsdp2-cp4、tp2-cp4、fsdp4-tp2、fsdp2-pp4、fsdp2-tp2-pp2、
fsdp2-tp4-ep8。成功 manifest 可续用；失败现场按时间戳归档。

## 3. 正式 eager-vs-graph 实验

跑通后，正式结论必须由同设备 eager single reference 与 graph candidate 比较：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

tests/glm5_2_graph_debug/run_graph_mode.sh inductor precision \
  --data --data-device npu --topology all
tests/glm5_2_graph_debug/run_graph_mode.sh inductor precision \
  --capture reference --topology all
tests/glm5_2_graph_debug/run_graph_mode.sh inductor precision \
  --capture candidate --topology all
tests/glm5_2_graph_debug/run_graph_mode.sh inductor precision \
  --compare --topology all --require-all
```

性能或精度+性能组合分别使用 `performance`、`combination` target，原有参数原样
透传。NPUGraphs 命令形式相同，但在移除 replay 降级前不应发布性能加速结论。

## 4. 问题、处理与三仓归属

### 4.1 文档与实现索引

本文是图模式调试状态的主报告；问题是否解决、采用何种降级、是否达到正式验收，均以
本文为准。三仓只记录各自职责内的实现，避免把实验结论分散到多个仓库：

| 仓库 | 入口 | 职责 |
|---|---|---|
| `torchtitan-test` | [图模式使用入口](README.md)、[原始调试入口](../glm5_2_graph_debug/README.md)、[报告索引](../glm5_2_graph_debug/experiments/reports/index.md)、[失败历史](../glm5_2_graph_debug/experiments/reports/failures.md) | 组织 eager/graph、single/all 拓扑实验，保存日志和报告，记录完整调试证据并给出验收结论。 |
| `TorchTitanTurbo` | [图模式 patch 说明](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/torchtitanturbo/tools/GRAPH_MODE.md)、[patch 清单](https://github.com/loveofguoke/TorchTitanTurbo/blob/glm-dev/PATCHES.md)、`torchtitanturbo/tools/graph_compat.py` | 实现默认关闭的 Ascend 专用兼容 patch；说明触发变量、patch 对象和后端限制。 |
| `torchtitan` | `torchtitan/distributed/compile.py` | 提供设备无关的 compile 配置和调用流程；不承载 CANN、HCCL、torch_npu 或 NPUGraph workaround。 |

问题级原始证据使用 `G001` 至 `G019` 编号保留在失败历史中；下表是面向当前代码和
验收状态的汇总。后续修复一个问题时，应同时更新原始问题条目、本文状态以及对应仓库
的实现说明。

状态含义：

- `已验证`：对应实现已在服务器目标环境和相关拓扑复验；
- `已实现，待复验`：代码和隔离测试已完成，但三仓当前版本尚未完成服务器回归；
- `降级通过，未根治`：训练可通过，但依赖 fallback、skip 或受限配置；
- `未根治`：后端根因仍存在，应用仓库没有可接受的最终修复。

### 4.2 当前问题状态

| 问题 | 当前处理 | 归属/状态 |
|---|---|---|
| CANN 9.0/9.1 混用 | Python 前 `env -i` + source CANN 9.1 | test；已验证 |
| HCCL 两类端口冲突 | NPU socket auto；host base port 每次生成 | test；已验证 |
| 冷编译超时/并发 | HCCL 600s、PG 2400s、每 rank 1 compiler worker | test；已验证，但属于调试稳定配置 |
| `aten.sum` UB tile 超限 | 最小 ACLNN fallback | test；降级通过，等待 torch_npu/Inductor 根治 |
| compiled all-reduce 数值异常 | 最小 collective fallback | test；降级通过，等待 torch_npu/Inductor 根治 |
| Turbo 覆盖 task queue | Turbo 尊重 `TORCHTITAN_TASK_QUEUE_ENABLE` | Turbo；已实现，待三仓版本复验 |
| TP `aten.complex` 无 strategy | 注册现有 broadcast pointwise strategy | Turbo；已实现，待三仓版本复验 |
| PP metadata batched P2P 损坏 | metadata 使用 `use_batch=False` | Turbo；已实现，待三仓版本复验 |
| EP 空 expert grouped-mm | 空组补零行、全空 bypass、显式 backward | Turbo；已实现，待三仓版本复验 |
| 动态 pointwise grid=0 | 两类 NPU Triton autotuner launch 前 no-op | Turbo；已实现，待三仓版本复验 |
| grouped-mm padding offsets 变为 INT64 | `cumsum` 显式保持 INT32 | Turbo；已实现，待三仓版本复验 |
| grouped-mm 无法 capture | AOT 兼容降级，replay skip | 后端限制；降级通过，未根治 |
| TP `DeviceMesh` runtime input 被拒绝 | AOT 兼容降级，replay skip | graph tree 限制；降级通过，未根治 |
| NPUGraph 降级模式下异步 queue 超时 | 兼容 profile 使用 queue 0，原生 capture 调试才使用 1 | test/Turbo；策略已验证，当前三仓集成待复验 |
| checkpoint 残留耗尽 streams | token-scoped 进程组外 worker 清理 | test checkpoint；已验证 |
| shell 脚本误交给 Python 编译检查 | shell 使用 `bash -n`，Python 文件使用 `py_compile` | test；已纠正，不属于后端问题 |

TorchTitan 保持设备无关，没有加入 CANN、HCCL、torch_npu 或 NPUGraph monkey patch。
NPU 兼容逻辑已从 test 根目录迁入
`torchtitanturbo/tools/graph_compat.py`；`train_npu.py` 只负责导入 Turbo 和注入实验级
process-group timeout。所有 Turbo patch 均由环境变量选择，普通 eager 不启用。
对应开发分支为 TorchTitanTurbo `glm-dev`；服务器复验前需要源码安装
该分支，并确认 `torchtitanturbo.__file__` 指向当前工作树。

```bash
cd /workspace/y50064852_yyb/TorchTitanTurbo
git switch glm-dev
python -m pip install -e . --no-deps
python - <<'PY'
import torchtitanturbo
print(torchtitanturbo.__file__)
PY
```

## 5. 修复实现与验证步骤

1. 逐项回放 `experiments/reports/failures.md`，确认错误发生层级和最小复现拓扑。
2. 保持 test 的 CANN/HCCL/cache/fallback 配置，不把实验参数硬编码进模型仓库。
3. 将 TP、PP、EP、Triton runtime 与 NPUGraph skip policy 迁入 Turbo 的独立 opt-in
   模块，保留原环境变量以兼容服务器脚本。
4. 修改 Turbo 环境初始化，使显式 task queue 不再被默认值 2 覆盖。
5. 将 test `train_npu.py` 收缩为薄 bootstrap，避免测试仓库长期维护 NPU 后端实现。
6. 增加 Turbo 单测，验证 task queue 优先级和 graph 布尔配置校验；运行 Python
   compile 与 diff whitespace 检查。
7. 服务器同步后应先执行 env、single、一个 TP、一个 PP、一个 EP、最复杂组合，再
   执行 all；每次保留 runtime log、manifest 和 invocation report。

推荐复验顺序：

```bash
cd /workspace/y50064852_yyb/torchtitan-test
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology single --force
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology tp8 --force
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology pp8 --force
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke --topology ep8 --force
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology fsdp2-tp4-ep8 --force
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke
tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke
```

## 6. 仍需继续解决的问题

- NPUGraph native replay：必须在 torch_npu/CANN graph tree 层解决 grouped-mm 捕获
  同步和非 Tensor `DeviceMesh` 输入 contract；应用层跳过 replay 不是最终方案。
- Inductor zero-fallback：需要修复 reduction tile 选择和 compiled collective 数值
  语义，之后关闭 fallback 重跑 precision，不能仅以 smoke 成功验收。
- 性能：当前 queue 0、compile thread 1 和 fallback 以稳定调试为目标，不能直接作为
  最优性能配置。
- 完整交付：仍需执行 5000-step eager-vs-Inductor 精度、profiler-off 性能重复、
  profiler-active 定位、checkpoint 与小时级 stability；NPUGraph 原生 replay 修复后
  同样重跑。
