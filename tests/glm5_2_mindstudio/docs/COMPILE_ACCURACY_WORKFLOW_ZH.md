# GLM-5.2 NPU eager/compile 官方精度流程

本文说明本仓库怎样使用 msProbe `PrecisionChecker` 检查同一 NPU 环境中
eager 与 `torch.compile` 的模块级前向/反向差异，以及该结果与正式图模式实验
之间的边界。

从 single/backend smoke 扩到 FSDP、TP、PP、EP 和 all topology 前，按
[服务器验收矩阵](SERVER_VALIDATION_MATRIX_ZH.md) 逐级验证。PrecisionChecker
single-pass 完成只证明被标记模块的官方比较阶段完成，不等价于整网 fullgraph 或
多 step 图训练精度通过。

官方依据：

- [PyTorch 编译精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md)
- [msProbe PrecisionChecker 源码](https://github.com/Ascend/msprobe/blob/master/python/msprobe/pytorch/compile_accuracy_checker/precision_checker.py)
- [PyTorch torch.compile 文档](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
- [Ascend Extension for PyTorch torch.compile 文档](https://www.hiascend.com/document/detail/zh/Pytorch/latest/userguide/torchcompile/docs/zh/torch_compile/pytorch_compilation_mode.md)

## 1. 为什么先做 NPU eager/graph，而不是 GPU/graph

GPU eager 与 NPU graph 同时混入两类变量：

```text
硬件/算子后端差异 + eager/compile 编译差异
```

因此标准拆分是：

```text
实验 A：GPU eager vs NPU eager
  → 验证设备迁移精度

实验 B：NPU eager vs NPU compile
  → 在同一硬件、CANN、torch_npu、checkpoint、token、拓扑下
    隔离图编译引入的差异
```

只有 A、B 都通过，才能把 NPU graph 与 GPU 基准连成完整证据链。

## 2. PrecisionChecker 的 single-pass 机制

### 2.1 数据流

训练/FSDP2 使用 single-pass：

```text
同一个训练进程、同一个 train step、同一个模块输入 x

                         ┌─ eager module(x) ─── eager output
x ─ PrecisionChecker ────┤
                         └─ compiled module(x) ─ compiled output
                                      │
                                      ├─ 比较 forward output
                                      └─ backward 时比较 input gradient

compiled 分支继续参与原训练流程、loss、optimizer
eager 重放只用于被标记模块的局部比较
```

它不是启动两个完整训练任务，也不是把整个模型复制两份训练。其优势是：

- 同一个输入直接进入两条模块路径；
- 不复制 optimizer 和训练状态；
- 不改变模型参数的所有权；
- 兼容禁止 `deepcopy` 的 FSDP2/DTensor 模型；
- 可以在已有训练 step 外安装和收集。

### 2.2 为什么 single-pass 的 loss_eager 是 NaN

single-pass 没有完整执行一条 eager 全模型链路，所以不存在“eager embedding →
所有 eager blocks → eager head → eager loss”。它只对被标记模块做局部 eager 重放。
因此报告中的 `loss_eager=NaN` 是预期行为，不是精度失败。

single-pass 的判据是：

- 被标记模块的 eager/compiled forward output；
- 对应 backward input gradient；
- 官方 CSV 中的误差与状态。

整网 eager/compiled 训练过程的 loss、grad norm 与关键状态由本标准流程的训练采集
阶段单独记录；模块级 checker 只负责缩小编译前后数值差异的首个发生范围，不能用
一张模块 CSV 代替整网训练结论。

### 2.3 为什么训练不使用 two-pass

two-pass 会构造 eager 模型和 compiled 模型的副本，各自跑完整链路。它更适合
普通单卡推理，因为能比较真实整网 output/loss，又没有 optimizer 状态污染。

但 FSDP2 把原始参数表示为分片 DTensor，模型不能安全 `deepcopy`。官方文档明确
要求 FSDP2 使用 single-pass。因此本项目训练 adapter 固定 `single_pass=True`。

## 3. 本项目接入点

### 3.1 只在选定训练 step 安装 checker

默认 dump step 是 msProbe 0-based `step0`，对应 TorchTitan 第一个训练 step。
adapter 把目标换算为 TorchTitan 的 1-based step：

```text
target_training_step = max(msprobe_dump_steps) + 1
```

在其他 step 中调用原始 `Trainer.train_step`；只在目标 step：

1. 为每个 pipeline model part 创建一个 `PrecisionChecker`；
2. 标记要编译/比较的模块；
3. `install(model_part)`；
4. 执行原始训练 step；
5. `collect(loss)`；
6. 每 rank、每 part 写官方 CSV；
7. 保留官方列并合并为每 rank CSV。

capture 结束前还会执行两类硬校验：

1. 每个 rank 的每个 `model_part` 必须实际包含至少一个
   `Glm5TransformerBlock`，并把期望 block 名称和数量写入
   `compile_coverage_rank<rank>.json`；
2. 每个 part CSV 必须至少有一条非 SKIP 的模块 `fwd_output`，以及一条
   `grad_input` 或 `grad_output`。只有表头或合成的 `LOSS` 行不能通过。

此外，所有 official capture 都会复用 precision 框架的运行时输入契约校验：
逐 local rank 核对 step 顺序、DP/PP rank、global token slots、sample SHA-256
和 replica 一致性。只有 `input_contract.valid=true` 才会发布 artifact manifest。

### 3.2 默认标记 GLM Transformer block

默认策略：

```python
checker.wrap_by_policy(model_part, (Glm5TransformerBlock,))
```

这使每个 GLM transformer block 成为独立诊断单元。好处是粒度稳定、报告可读，
并能覆盖 attention、DSA indexer、MoE 和残差组成的完整 block。

另一种配置是 `children`，按模型子模块深度标记。CLI 已显式暴露 policy/depth。
下面展示完整的新 identity 流程：

```bash
COMMON="--compile-policy children --children-depth 2"

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --data --data-device npu --topology single ${COMMON}
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology single ${COMMON}
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology single ${COMMON}
```

这些参数会进入 identity。不能临时改运行时 JSON。

### 3.3 pipeline model parts

PP/ZBPP 下一个 rank 可能拥有一个或多个 virtual stage。adapter 为每个
`model_part` 建立 checker，并在模块名前加 `partN.`，避免不同 stage 同名节点
在 rank CSV 中碰撞。

## 4. 标准命令

所有命令从 `torchtitan-test` 根目录执行。

### 4.1 单卡

```bash
export ASCEND_RT_VISIBLE_DEVICES=4

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --data --data-device npu --topology single

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology single

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology single
```

compile workflow 没有单独的 `--capture reference`，因为 eager reference 已在
candidate 的同一次 single-pass 中产生。尝试 reference capture 应明确报错。

### 4.2 指定一个分布式拓扑

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology fsdp8

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology fsdp8
```

### 4.3 all topology

```bash
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --list-topologies

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology all

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology all
```

`all` 会展开当前注册且 world size 不超过 8 的拓扑。不同 topology 可能触发
不同的 compile/DTensor/collective 支持问题，所以先完成：

```text
single → 一个 FSDP/TP/PP/EP 代表拓扑 → all
```

接口支持不等于当前 CANN/torch_npu/msProbe 组合已验收所有拓扑。

### 4.4 backend 与 graph dump

默认：

```bash
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology single --backend inductor
```

关闭 checker graph dump：

```bash
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology single --no-dump-graphs
```

不保存 checker 重放输入：

```bash
python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology single --no-capture-input
```

这两段只展示参数位置。`--no-dump-graphs`、`--no-capture-input`、`--backend`、
`--compile-policy`、`--children-depth` 都会改变 identity，正式运行时必须把同一组
参数附到 data、capture 和 compare 三个阶段。

backend 名称是否存在可先检查：

```bash
python - <<'PY'
import torch
import torch_npu

print(torch.compiler.list_backends())
PY
```

列表中存在只说明已注册，不说明 GLM 训练、反向、分布式和当前动态 shape 全部
支持。每个 backend 都要从单卡 smoke 开始。

## 5. `fullgraph=False` 与生产 fullgraph 的边界

官方 PrecisionChecker 的目标是逐模块诊断。当前 msProbe 源码路径内部采用
`torch.compile(..., fullgraph=False)`；某个模块内部遇到不支持捕获的语义时，
Dynamo 可以断图并继续。这有助于拿到局部精度报告，但它不能证明生产训练满足：

```text
整块 fullgraph=True，无 graph break，无意外 eager fallback
```

TorchTitan 的正式 block compile 路径使用更严格的 fullgraph 语义。因此：

| 检查 | 回答的问题 |
| --- | --- |
| PrecisionChecker | 被标记模块中 eager 与 compiled 的前向/反向数值是否一致 |
| graph smoke | 所选 backend 和拓扑能否跑通 |
| `fullgraph=True` 正式图实验 | block 能否完整捕获，不允许隐式断图 |
| TORCH_TRACE/tlparse | 哪里断图、guard 失败、重编译，生成哪些 FX/IR/code |
| graph precision | 多 step 整网 loss/grad norm 是否对齐 |
| graph performance | 图模式是否真正减少 launch、提高吞吐，是否有编译摊销 |

不能用 PrecisionChecker CSV PASS 直接宣布“图模式完整入图并加速”。

## 6. 不允许双重编译

运行 PrecisionChecker 时不要同时开启 TorchTitan 原生 `--compile.enable`：

```text
错误组合：
  TorchTitan 先 compile block
    + PrecisionChecker 再 wrap/compile 同一 block
```

双重编译会改变模块边界、hook 对象和 graph cache，结果不再是官方 checker 预期
的 eager/compiled 对照。因此同一次 checker 运行只能由 checker 管理被测模块的
编译边界；生产整网编译应作为另一个标准采集阶段运行，再把两阶段证据关联到同一
实验身份中，不能在一个 Python 进程里叠加两套编译入口。

## 7. 输出与汇总

每个 rank 的原始输出类似：

```text
mindstudio_artifacts/<experiment>/<topology>/candidate-r1/official/
├── precision_rank0_part0.csv
├── precision_rank0.csv
├── compile_coverage_rank0.json
├── precision_rank1_part0.csv
├── precision_rank1.csv
├── compile_coverage_rank1.json
└── graphs/
    ├── part0/
    └── ...
```

compare 阶段不再运行模型，而是把各 rank CSV 合并为：

```text
mindstudio_reports/<experiment>/<topology>/official_compare/
├── compile_all_ranks.csv
├── official_summary.json
└── runtime.log
```

官方说明 FSDP2 场景每 rank 独立写 CSV，工具本身不内置多卡聚合。本项目的
aggregate 只增加 scenario/rank/part 语义并保留官方字段，不重新计算官方数值。

## 8. CSV 指标与判读

不同 msProbe revision 的列名可能变化，应以锁定源码和实际表头为准。核心关注：

- 模块名；
- forward output 差异；
- backward/input gradient 差异；
- 最大绝对差、最大相对差或官方对应字段；
- `all_pass`/状态；
- loss_compiled 与 single-pass 下预期的 loss_eager NaN；
- warning、skip、unavailable 等非硬失败状态。

`compile_coverage_rank<rank>.json` 不是 msProbe 自带指标，而是本项目的覆盖证据：
它记录 policy、每个 model part 中实际发现的 GLM block、对应 part CSV，以及该
CSV 中决定性的 forward/backward 行统计。对默认 `glm5-block` 策略，block 是诊断
粒度；对 `children` 策略，期望 block 列表只证明该 part 属于 GLM block 数据流，
实际被 checker 标记的是配置深度的 children，不能把两组名称强行一一对应。

### 8.1 当前 checker 数值容差

当前接入不自定义阈值，遵循所锁定 msProbe revision 的实现。当前审阅的官方
源码使用固定 `torch.allclose` 语义：

\[
|a-b| \leq \operatorname{atol}+operatorname{rtol}\times |b|,
\]

其中实现口径为 `atol=1e-4`、`rtol=1e-3`。每次升级 msProbe 后必须重新核对
源码，不能把这组值当成永不变化的标准。

配置或 manifest 中记录的说明字段不是给官方 checker 注入阈值。报告必须记录并解释
官方 checker 实际执行的容差，不能用另外一组阈值改写官方 Result。

### 8.2 不要盲信一个 ALL PASS

当前官方实现存在需要保守解释的状态语义：

- single-pass 的 loss_eager NaN 不应判失败；
- warning/skip 可能不把 `all_pass` 置为 false；
- 某些汇总字段不包含 loss 差异；
- checker 之外的模块没有被检查；当前硬校验保证每个 rank/part 有非零 GLM block
  和决定性的前反向结果，但不声称整张模型的每一个子模块都被逐项比较；
- graph break/fallback 后局部仍可能得到数值结果。

因此项目 summary 必须同时统计 hard fail、warning、skip、unparsed，并保留原始
CSV。只有“每个 rank/part 的非零覆盖与决定性前反向证据成立 + 无 hard fail +
warning/skip 已解释 + 正式图实验通过”才是可交付结论。

## 9. 分布式场景逐类理解

### 9.1 FSDP2

- 参数是 sharded DTensor，不能 deepcopy；
- 使用 single-pass；
- 每个 rank 检查自己的 local shard/局部执行；
- 汇总要保留 rank，不能先平均掉某个 rank 的失败；
- checker PASS 后仍要验证 collective、reshard 和整网 loss。

### 9.2 TP

每 rank 的 block 输入/参数可能是 Shard 或 Partial placement。局部 eager/compiled
一致只能说明 rank-local 数值一致；还要检查 DTensor redistribution 和 collective
是否被完整捕获。若只有某 rank 失败，应结合 local shape、placement 和 collective
调用栈，而不是只看全局模块名。

### 9.3 PP

不同 rank 拥有不同 stage。报告中的 `partN.` 是 rank 内 virtual stage，不等于
global layer 编号。定位时需要结合 pipeline partition 映射。只有最后 stage 有完整
loss，single-pass 本来也不提供整网 eager loss。

### 9.4 EP

不同 rank 收到的 token/expert 路由不同。离散 top-k 边界变化可能导致两条局部
路径选到不同 expert；此时先确认同一模块输入、router score、top-k 和 dispatch
映射。不能把 expert 输出无法自动配对简单归因于 kernel 精度。

### 9.5 组合拓扑

FSDP+TP+PP/EP 等组合同时包含参数 shard、激活 shard、stage ownership 和 token
dispatch。报告必须以 `(topology, rank, part, module)` 为最小定位键，不能只按
module name 合并。

## 10. 编译异常的标准下钻

当 compile CSV 出现异常时：

```text
1. 确认同一输入的 eager/eager 自检正常
2. 找到第一个失败 block/rank/part
3. 查看 checker graph dump
4. 用 TORCH_LOGS="graph_breaks,recompiles,dynamic" 重跑最小图 smoke
5. 用 TORCH_TRACE + tlparse 看 graph、guard 和 recompile
6. 区分 Dynamo graph break 与 backend lowering/fallback
7. 修复或建立显式稳定边界
8. 重跑 checker
9. 重跑相同训练区间的 eager/compile 数值比较
10. 关闭精度工具，运行无采集基线和 profiler-active 性能 A/B
```

性能证据按 [PERFORMANCE_WORKFLOW_ZH.md](PERFORMANCE_WORKFLOW_ZH.md) 采集并导入
MindStudio Insight；编译精度与性能使用相同源码、输入和拓扑，但保留独立 capture。

## 11. 图断裂、重编译和 fallback 的区别

### 11.1 graph break

Dynamo 遇到无法安全表示为 FX Graph 的 Python 语义，结束当前 graph，eager 执行
中间片段，再继续捕获。后果是跨边界融合丢失、launch 增加，但 `fullgraph=False`
可能仍跑完。

### 11.2 recompile

FX Graph 已成功捕获，但输入 shape、dtype、对象 identity 等 guard 失败，Dynamo
为新条件重新编译。它不一定断图，但可能导致编译时间和 cache 膨胀。

### 11.3 backend unsupported/fallback

FX Graph 已捕获，但 Inductor/NPU backend 对某个 op 没有 decomposition、lowering
或 kernel，可能报错或回退。这不是 Dynamo graph break。解决方式是补 decomposition、
lowering/kernel 或调整明确的 backend 边界，而不是用断图掩盖。

PrecisionChecker 的数值比较不自动替用户完成这三类归因。

## 12. 已知限制和验收清单

已知限制：

- 当前只实现 single-pass 训练 adapter；
- 默认只 wrap `Glm5TransformerBlock`；
- backend 必须由服务器实际验证；
- checker `fullgraph=False`；
- 官方多 rank 结果由项目层汇总，不能丢失 rank 状态；
- warning/skip 需要人工解释；
- 该流程不等价于整网 eager/graph 多 step 精度；
- 工具升级可能改变内部 Dynamo API、CSV schema 和容差。

正式结论前检查：

- [ ] fixture generation、checkpoint、token plan 相同；
- [ ] 同一 NPU 软件栈和相同 topology；
- [ ] 没有同时开启 TorchTitan 原生 compile；
- [ ] 每个 rank/part 的 `compile_coverage_rank<rank>.json` 存在且 GLM block
  数量非零；
- [ ] 每个 rank/part CSV 存在；
- [ ] 每个 part 至少有非 SKIP 的模块 forward 和 backward 决定性行；
- [ ] 不是仅凭 CSV header 或 `LOSS` PASS 行得出结论；
- [ ] artifact manifest 中 `input_contract.valid=true`；
- [ ] hard fail 为 0；
- [ ] warning/skip 已逐项解释；
- [ ] `loss_eager=NaN` 按 single-pass 语义处理；
- [ ] fullgraph graph smoke/diagnostics 通过；
- [ ] 多 step loss/grad norm 通过；
- [ ] profiler-off 的性能与编译摊销符合预期。

只有这些证据一起成立，才能说图模式在该设备、软件栈、模型配置和拓扑下完成
精度验收。
