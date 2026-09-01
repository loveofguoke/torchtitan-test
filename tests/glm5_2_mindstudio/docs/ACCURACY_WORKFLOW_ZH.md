# GLM-5.2 MindStudio 官方精度迁移流程

本文说明如何按 MindStudio 26.1 官方训练精度指南完成 CheckList、问题复现、状态监测、
模块/API 采集、预检、比较、分级可视化和修复后验收。

官方依据：

- [MindStudio 文档总入口](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [msProbe PyTorch 精度数据采集](https://github.com/Ascend/msprobe/blob/master/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)
- [msProbe PyTorch 精度比对](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [msProbe PyTorch 精度预检](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_checker/pytorch_accuracy_checker_instruct.md)
- [msProbe PyTorch 快速入门](https://github.com/Ascend/msprobe/blob/master/docs/zh/quick_start/pytorch_quick_start.md)
- [大模型训练精度定位官方实践](https://www.hiascend.com/document/detail/zh/mindstudio/latest/practicalcases/LargeModelTrainingAccuracy/docs/zh/best_practices/train_debug_guide.md)
- [msProbe 源码](https://gitcode.com/Ascend/msprobe)

版本非常重要。本文描述本仓库 adapter 所依赖的当前官方接口，正式实验仍要把
msProbe revision、安装版本和 CANN/PyTorch/torch_npu 版本写入 artifact。

## 1. 官方精度定位的三层证据

### 1.1 现象层：训练结果是否超出交付标准

先在相同 checkpoint、数据、拓扑和训练配置下观察多 step loss 与 grad norm。它回答：

- GPU/NPU 的训练趋势是否一致；
- 绝对误差、相对误差和误差分布是否满足正式标准；
- 重复运行是否稳定；
- 差异是否随 step 累积或突然扩大。

loss 是整网压缩后的标量，只能确认问题现象，不能告诉误差第一次在哪个模块出现。

### 1.2 模块/API 层：偏差首先出现在哪里

msProbe 在前向和反向运行中采集 Module 或 PyTorch API 的输入、输出、参数、
梯度及统计量，再用官方 compare 比较 GPU/NPU。它回答：

- 是 embedding、attention、MoE、norm 还是 lm head 首先偏离；
- 偏差在模块输入已经存在，还是由当前模块输出新增；
- dtype、shape、`requires_grad` 或非 tensor 参数是否不一致；
- 偏差只存在于统计量，还是完整 tensor 已明显不同；
- 多 rank 中是否只有特定 rank 异常。

这层是正式定位入口，但它本身也可能通过 hook、同步和落盘改变运行时行为，
所以不使用 profiler-on/dump-on 的 step time 或 loss 作为真实性能/精度基线。

### 1.3 算子层：模块偏差由什么实现造成

当 L1/tensor 已把问题缩小到 API 后，用 stack 和输入 shape 回到具体算子实现，并按
官方指南选择 API 预检、单算子复现、Profiler、msSanitizer 或 msOpProf。模块输出异常
只是定位信号；最终根因需要“相同输入、当前算子输出新增异常”的证据。

## 2. 官方五阶段与当前落地状态

msProbe 快速入门给出的标准精度调试顺序是：

```text
1. 训练前配置检查
2. 训练状态监控
3. 精度数据采集
4. 精度预检
5. 精度比对
```

当前仓库的落地边界如下：

| 阶段 | 目的 | 当前实现 |
| --- | --- | --- |
| 配置检查 | 找出两端 seed、dtype、优化器、模型、环境等差异 | 已接入 dynamic `ConfigChecker` 和逐 rank 官方 compare |
| 训练状态监控 | 监控激活、梯度、权重、优化器及异常状态 | 已由 `training_monitor_benchmark.py` 接入 Monitor V2 |
| 数据采集 | L0/L1/mix，statistics/tensor | 已由 `migration_benchmark.py` 接入 `PrecisionDebugger` |
| 精度预检 | 对单端 API 构造单测、比较 CPU 高精度标杆，再比较 GPU/NPU 预检结论 | 已接入 `--precheck`、`--precheck-compare`，逐 step、逐 rank 保存官方结果 |
| 精度比对 | GPU/CPU golden 与 NPU target 比较 | 已由 `--compare` 调用官方 `msprobe compare` |

因此当前自动闭环是“固定输入 → 配置检查 → Monitor V2 长稳筛查 → capture →
API pre-check → official compare → 分级图可视化 → 中文索引”。Monitor V2 只给
逐 rank CSV 证据，不定义 GPU/NPU 自动 PASS/FAIL；可疑 step 仍需回到 dump 与
compare 精确定位。

## 3. 本项目如何接入 PrecisionDebugger

本仓库不修改 TorchTitan Trainer 源码。capture 进程启动前安装一个实验期 patch：

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all(seed=seed, mode=deterministic, rm_dropout=False)
debugger = PrecisionDebugger(config_path=config_path)

def train_step_with_msprobe(self, data_iterator):
    debugger.start(model=self.model_parts)
    try:
        return original_train_step(self, data_iterator)
    finally:
        debugger.stop()
        debugger.step()
```

这段逻辑位于 `capture_training.py`，只存在于 MindStudio capture 子进程：

1. `seed_all` 固定官方工具管理的随机性；
2. `start(model=self.model_parts)` 在一次 TorchTitan train step 前开启 hook；
3. 原始 `Trainer.train_step` 完整运行 forward、loss、backward 和 optimizer；
4. `stop()` 结束当前采集；
5. `step()` 落盘并把 msProbe 内部 step 推进一位；
6. 无论训练成功还是异常，`finally` 都尝试关闭当前采集。

模型数学实现仍来自 TorchTitan；NPU 端在导入 Trainer 前加载 TorchTitanTurbo。
同一进程还安装 fixed token dataloader，使两端消费相同 token。

### 3.1 step 编号

msProbe 的第一个 debugger iteration 是 `step0`。TorchTitan 日志通常把第一个
训练 step 打印为 `step: 1`。因此：

```text
--dump-step 0
  ↔ msProbe output/step0
  ↔ TorchTitan 第一个 train step
```

不要把 msProbe `step0` 误认为 seed checkpoint 生成过程。

## 4. 数据采集配置

### 4.1 L0、L1 与 mix

| level | 粒度 | 适合的问题 | 主要代价 |
| --- | --- | --- | --- |
| `L0` | `nn.Module` 模块级 | 首轮定位哪一层/哪个大模块开始异常 | 体积较小，细节有限 |
| `L1` | PyTorch API 级 | 已知模块后定位 matmul、norm、scatter、collective 等 API | hook/数据量明显增加 |
| `mix` | L0 + L1 | 模块结构和内部 API 一起看 | 体积、运行扰动最大 |

推荐顺序是 L0 → 缩小 scope → L1，而不是第一次就全模型 mix。

### 4.2 statistics、tensor 与校验模式

本 benchmark CLI 当前直接暴露：

- `statistics`：只保存 max/min/mean/L2norm 等摘要，适合第一轮；
- `tensor`：同时保存真实 tensor，能计算余弦、欧氏距离等完整指标，体积大。

配置对象和 CLI 还支持 `summary_mode=statistics|md5`、scope、module/API list、
tensor list、data mode、同步/异步 dump 和 extra info。例如下面这组 override
表示采 step0/step2、只采 rank0、使用 md5 摘要：

```bash
--dump-steps 0,2 --dump-ranks 0 --summary-mode md5
```

把这组参数原样附到 `--data`、`--capture reference`、`--capture candidate` 和
`--compare` 四条命令。所有 CLI override 都进入 experiment identity，少带一次就会
指向另一个目录，而不是复用旧数据。`--scope`、`--module-or-api`、`--tensor-list`
可重复，具体条目必须来自所锁定 msProbe revision 的 config schema，本文不虚构
模型名语法。不要临时手改某一端生成的 `msprobe_config.json`，否则 manifest 表示
的实验配置将失真。当前 summary mode 没有暴露 xor；要新增它应先核对目标版本
schema 并补配置验证和测试。

官方文档还列出 `acc_check`、`structure`、`overflow_check`、`nan_check` 等任务。
当前 `MsProbeDumpConfig` 没有宣称实现这些模式；需要时先按锁定版本新增显式
配置、测试和报告解析。

### 4.3 rank、step 和范围

默认 `rank=[]` 把 rank 选择交给官方语义，实际产物必须检查
`stepN/rankM/`。多卡 tensor dump 会近似按 rank 数放大。

估算 raw 体积时至少考虑：

```text
拓扑数 × 采集 rank 数 × dump step 数 × 模块/API 数 ×
(forward 输入输出 + backward 输入输出 + 参数/梯度)
```

先用一个 step、statistics、L0。只有结果指向某个模块后才用 tensor/L1 下钻。

## 5. 标准运行流程

以下命令均从 `torchtitan-test` 根目录执行。

### 5.1 生成固定实验输入

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device cuda --topology single
```

也可以在 NPU 端生成：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device npu --topology single
```

数据阶段生成并校验：

- 固定 token plan；
- 随机初始化 seed checkpoint；
- token/checkpoint hash；
- fixture generation ID；
- 生成命令和日志。

msProbe 不负责生成训练数据。它只观察使用这份契约的训练。

### 5.2 GPU reference

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology single
```

### 5.3 NPU candidate

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology single
```

两端 capture 可以在不同服务器执行。同步时保留完整 artifact 相对目录，不能只
复制一个 `dump.json` 后丢失 manifest、工具版本和附件 hash。

### 5.4 compare

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single
```

adapter 解析所选 step/rank 的官方输出，再运行等价于：

```bash
msprobe compare \
  -tp <candidate-npu-dump> \
  -gp <reference-gpu-dump> \
  -o <official-compare-output>
```

这里 target 是 NPU candidate，golden 是 GPU reference。不要反过来，否则结果
列名和 relative error 分母语义会被误读。

可选定位参数：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single --diff-analysis
```

真实 tensor compare 还可以输出单模块/API 日志或 XLSX：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single --tensor-log --xlsx
```

结构或名字无法自动配对时可传官方映射：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single \
  --data-mapping /absolute/path/data_mapping.yaml

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single \
  --cell-mapping /absolute/path/cell_mapping.yaml
```

映射不是“让不一致变成一致”。它只声明两端哪个数据/模块语义对应，文件必须
进入实验记录并由人工审阅。当前 adapter 按官方约束在启动前拒绝以下组合：

- fuzzy match 与 data mapping 同时开启；
- data mapping 用于一次包含多个 rank 的 compare；应通过 `--dump-ranks N`
  逐 rank 建立相同 identity；
- cell mapping 用于 L1/mix；它只支持 L0 module dump；
- `--tensor-log` 用于 statistics artifact；它只支持 tensor dump。

### 5.5 分布式与 all

接口复用公共拓扑注册表：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py --list-topologies
```

一个分布式例子：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology fsdp8

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology fsdp8

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology fsdp8
```

全拓扑：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology all
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology all
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology all
```

需要在各自 capture 前 export 对应的 8 卡可见设备。`all` 只代表 orchestrator
会展开注册表，不证明某个 msProbe/CANN revision 已通过每种拓扑。正式 all 前：

1. 单卡 statistics/L0；
2. 一个代表性分布式 topology；
3. 检查每 rank 文件与总体体积；
4. 再扩大范围。

## 6. 官方输出结构

典型 msProbe dump：

```text
official/
└── step0/
    ├── rank0/
    │   ├── dump.json
    │   ├── stack.json
    │   ├── construct.json
    │   ├── dump_error_info.log       # 仅错误时可能出现
    │   └── dump_tensor_data/          # tensor task 时
    ├── rank1/
    └── ...
```

文件意义：

- `dump.json`：Module/API 名称、前后向、dtype、shape 和统计/校验数据；
- `stack.json`：调用栈，用于回到 TorchTitan/Turbo 代码；
- `construct.json`：模块层级；L1 纯 API 模式可能为空；
- `dump_tensor_data/*.pt`：真实输入、输出、参数和梯度；
- `dump_error_info.log`：dump 工具错误，不是模型精度结论。

官方文档说明，tensor 文件可能在算子执行后逐步落盘，而 JSON 需要
`PrecisionDebugger.stop()` 后才完整。异常退出后存在 `.pt` 不代表 capture 完整，
本项目必须等校验和 `complete.json` 成功后才允许 compare。

## 7. 指标与公式

设 NPU tensor 为 \(N\)，golden tensor 为 \(B\)，元素数为 \(n\)。

### 7.1 真实 tensor 模式

逐元素相对误差：

\[
RE_i = \left|\frac{N_i-B_i}{B_i}\right|.
\]

余弦相似度：

\[
\operatorname{Cosine}(N,B)=
\frac{N\cdot B}{\|N\|_2\|B\|_2}.
\]

欧氏距离：

\[
\operatorname{EucDist}(N,B)=\|N-B\|_2.
\]

最大绝对误差：

\[
\operatorname{MaxAbsErr}(N,B)=\max_i |N_i-B_i|.
\]

最大相对误差：

\[
\operatorname{MaxRelativeErr}(N,B)=\max_i RE_i.
\]

千分之一、千分之五误差带内比例：

\[
R_{0.001}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(RE_i<0.001),
\qquad
R_{0.005}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}(RE_i<0.005).
\]

官方经验说明 Cosine 越接近 1 越好，常见参考为大于 0.99；MaxAbsErr 越接近
0 越好，常见参考为小于 0.001。但这不是所有 BF16 模块都可机械套用的项目
验收线。还要看 dtype、数值尺度、输入误差、模块类型和下游影响。

当 golden 元素为 0 或 tensor 含 NaN 时，相对误差可能是 `inf`/`nan`。此时要
结合绝对误差和有效元素分布，不能只看 MaxRelativeErr。

### 7.2 statistics 模式

只比较摘要。例如：

\[
\Delta_{\max}=\max(N)-\max(B),
\]

\[
\Delta_{\operatorname{mean}}=\operatorname{mean}(N)-
\operatorname{mean}(B),
\]

\[
\operatorname{MeanRelativeErr}=
\left|\frac{\operatorname{mean}(N)-\operatorname{mean}(B)}
{\operatorname{mean}(B)}\right|\times100\%,
\]

\[
\operatorname{NormRelativeErr}=
\left|\frac{\|N\|_2-\|B\|_2}{\|B\|_2}\right|\times100\%.
\]

statistics 很适合筛选，但不同 tensor 可能具有相同 max/min/mean/L2norm，
因此 statistics PASS 不能证明逐元素一致。可疑节点需要 tensor 模式复查。

### 7.3 Result 的含义

官方结果分为 `pass`、`warning`、`error`，优先级是：

```text
error > warning > pass
```

结果还可能包含无法配对、unsupported 或工具异常。项目 summary 应保守地把
这些状态暴露出来，而不是把“没有 error”自动改写为 PASS。

常见 error 原因包括：

- NPU 最大/最小出现 golden 没有的 NaN/Inf；
- 输入误差较小而输出误差显著放大；
- `requires_grad` 不一致；
- 非 tensor 标量参数不一致；
- 校验值不一致；
- dtype 或 shape 不一致。

`Err_Message` 比单个数值更重要，因为它说明这一行为什么被分级。

## 8. 结果阅读方法

不要从最后一行往前猜。按数据流做以下检查：

1. **配对完整性**：NPU Name 和 Bench Name 是否真的代表同一模块/API；
2. **结构契约**：dtype、shape、requires_grad、参数是否一致；
3. **输入基线**：当前模块 input 是否已偏离；
4. **新增误差**：input 接近但 output 明显放大，当前模块更可疑；
5. **前向/反向**：只在 backward 放大时检查梯度 kernel、reduction 和通信；
6. **跨 rank**：异常是所有 rank 同步出现，还是某一 rank 的数据/路由/通信问题；
7. **数值尺度**：绝对误差是否只是 BF16 在该尺度的一档 ULP；
8. **下游影响**：误差是否导致 loss/grad norm、离散 top-k 或最终输出异常；
9. **调用栈**：用 stack.json 回到 TorchTitan、Turbo 或 torch_npu；
10. **最小复现**：缩小到一个模块/API 后再开 tensor 和更深 trace。

### 8.1 一个具体例子

假设结果是：

```text
attention_norm.input:  pass
attention_norm.output: pass
wq_a.input:            pass
wq_a.output:           error, NormRelativeErr suddenly grows
rope.input:            already different
```

合理结论不是“rope FAIL，所以 rope 有 bug”。数据流显示差异在 `wq_a.output`
首次显著扩大，rope 只是消费了已经偏离的输入。下一步应该：

1. 用 L1 采集 `wq_a` 内部 linear/matmul/cast；
2. 查看 dtype、shape 和具体 tensor；
3. 对照 GPU/NPU kernel 路径；
4. 用相同输入做 API 预检或最小单算子复现；
5. 验证修复后 `wq_a.output` 和后续 loss 是否同时改善。

## 9. 配置检查、状态监控与预检的边界

### 9.1 配置检查

配置检查应该在数值 dump 前回答：

- 两端模型配置和参数数量是否一致；
- seed、deterministic、dtype、mixed precision 是否一致；
- optimizer、scheduler 和 loss scaling 是否一致；
- batch、sequence、token plan、checkpoint 是否一致；
- 分布式 degree、rank、world size 是否一致；
- 关键环境变量和软件版本是否存在差异。

发现配置不一致时先修配置。用大量 tensor dump 解释一个不同 checkpoint 没有意义。

当前 adapter 在训练进程中调用 `ConfigChecker.apply_patches("pytorch")`，然后以
实际模型和解析后的启动脚本为输入，为每个 global rank 生成：

```text
official/config_check_rank0.zip
official/config_check_rank1.zip
...
```

离线阶段逐 rank 执行：

```bash
msprobe config_check \
  -c <gpu-reference-rankN.zip> <npu-candidate-rankN.zip> \
  -o <report/rankN>
```

完整命令：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --data --data-device cuda --topology single
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --capture reference --topology single

export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --capture candidate --topology single

python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --compare --topology single
```

分布式时不能只看 rank0。每个 rank 可能拥有不同 PP stage、DTensor shard 或环境，
所以项目逐 rank 生成包并逐 rank compare，再由总报告索引这些结果。

### 9.2 训练状态监控

官方训练状态监控与本项目 raw metrics 目的不同：前者面向运行时计算、通信、
优化器等异常，后者记录 loss/grad norm 作为端到端数值证据。当前
`training_monitor_benchmark.py` 在 Trainer 创建模型与优化器后调用
`TrainerMonitorV2.start()`，每个完整 train step 后调用一次 `step()`，并在正常
结束或异常退出时通过 `finally` 调用 `stop()`。默认仅开低开销 `weight_grad`；
module/optimizer/param/cc 是显式可选项。完整命令与 PP/compile 边界见目录 README。

```text
TorchTitan Trainer.train
  ├─ TrainerMonitorV2.start(model, optimizer, grad_acc_steps)
  ├─ train_step → optimizer step → monitor.step()  # 恰好一次
  ├─ ...
  └─ finally → monitor.stop()
```

配置固定 `patch_optimizer_step=false`，否则官方自动 patch 与项目的显式 `step()` 会
重复计数。single 最小闭环：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --data --data-device cuda --topology single
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture reference --topology single --training-steps 100

unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture candidate --topology single --training-steps 100

python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --compare --topology single --training-steps 100
```

每个端点的 `official/rank_<rank>/**/*.csv` 是权威监控数据；compare 只生成
`official_compare/monitor_index.json` 将两端身份关联起来。官方 Monitor V2 没有
GPU/NPU cross-device 数值 comparator，所以这里的 `unparsed`/索引完成不是 FAIL，
也不能改写成 PASS。先从 CSV 找出异常 step/rank/module，再在对应 step 运行 L0/L1
dump 与多 step precision。

分布式先验证 FSDP/TP/EP 的 sharded 参数与 optimizer 容器，再验证 PP 的 model-parts
名称和 optimizer ownership。当前 workflow 会警告 PP 尚需服务器专项验收，并拒绝
Monitor V2 与正式 `--compile.enable` 组合，避免静默产生不可解释数据。

### 9.2.1 分级图可视化

分级图不是重新 capture。它读取同 generation、同 L0/mix 配置的 GPU/NPU
`construct.json`，调用官方 `graph_visualize` 生成 `.vis.db`：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --graph-visualize --topology single --level L0

tensorboard \
  --logdir mindstudio_artifacts/<experiment-id>/single/graph-visualize-r1/official \
  --bind_all
```

服务器只负责生成数据库并启动 TensorBoard；通过 VSCode 端口转发在本地浏览器打开
Ascend Graph 插件。可用 `--fuzzy-match`、`--graph-overflow-check`、
`--graph-progress-log` 控制官方匹配和标注。tracked report 只保存 `.vis.db` 的相对
路径、SHA-256、runtime log 和启动命令。Release `analysis` 会保留处理后的数据库，
但上传前仍要审查模块名、统计量和源码/服务器路径。

### 9.3 precision pre-check

pre-check 不是对同一条网络执行结果再算一组统计量，而是把 L1/mix capture 中
记录的 API、shape、dtype 和数据特征交给官方 checker，为每个 API 构造可复现的
单元测试：

```text
GPU L1/mix dump.json ── acc_check ── GPU vs CPU高精度 details ┐
                                                               ├─ api_precision_compare
NPU L1/mix dump.json ── acc_check ── NPU vs CPU高精度 details ┘
```

`statistics` dump 主要依据统计信息构造输入，成本较低，适合初筛；`tensor` 可使用
真实输入，定位更可信，但数据量、敏感性和耗时更高。预检只支持 API 级 capture，
因此当前 adapter 会拒绝 L0 artifact。

先用同一组 override 完成 data、GPU/NPU capture：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device cuda --topology single \
  --level L1 --dump-task statistics
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology single \
  --level L1 --dump-task statistics

export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology single \
  --level L1 --dump-task statistics
```

再在两端分别执行 checker：

```bash
# GPU server
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck reference --topology single \
  --level L1 --dump-task statistics

# NPU server
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck candidate --topology single \
  --level L1 --dump-task statistics
```

默认调用单 device 的 `msprobe acc_check`。API 数量大时，传
`--precheck-splits N` 改用 `multi_acc_check`，并用
`--precheck-device-ids 0,1` 选择离线单测使用的 device。这里的 device 只服务于
checker，不改变原始模型 capture 的训练拓扑。

自动 `--precheck-compare` 要求每个端点、step、rank 恰好一个 details CSV。
multi-device 预检可能产生多份 details；项目会完整保存并在 manifest 标记为不兼容
自动 compare，而不是任意挑一份。正式自动闭环优先用单 device；若确需多 device，
应按照锁定版本官方规则明确归并/选择流程后再手工比较。

`acc_check` 中途失败且已有成对的
`accuracy_checking_result_<timestamp>.csv`/`accuracy_checking_details_<timestamp>.csv`
时，可使用官方断点续检：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck candidate --topology single \
  --level L1 --dump-task statistics --dump-step 0 --dump-ranks 0 \
  --precheck-resume-csv /absolute/path/accuracy_checking_result_<timestamp>.csv
```

为防止一个 CSV 被错误用于多个成员，wrapper 要求整个 action 恰好选择一个
topology/step/rank。它校验同名 details 文件存在，把两份源 CSV 复制到当前 member
的 `official/` 后再续写；源文件不会被修改。源路径与 SHA-256 进入 operation
identity。resume 只解决同一 dump 的工具中断，不允许跨 capture/config 复用。

把两端完整 artifact 和下列结果同步到同一环境：

```text
mindstudio_artifacts/<experiment-id>/<topology>/precision_precheck/reference-r1/
mindstudio_artifacts/<experiment-id>/<topology>/precision_precheck/candidate-r1/
```

然后执行：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck-compare --topology single \
  --level L1 --dump-task statistics
```

项目对每个配置的 step/rank 单独保存：

```text
mindstudio_artifacts/<experiment-id>/<topology>/
precision_precheck/<role>-r1/step0/rank0/
├── official/
│   ├── accuracy_checking_result_<timestamp>.csv
│   └── accuracy_checking_details_<timestamp>.csv
├── invocation.json
├── runtime.log
├── manifest.json
└── complete.json

mindstudio_reports/<experiment-id>/<topology>/
precision_precheck/compare-r1/step0/rank0/official/
├── api_precision_compare_result_<timestamp>.csv
└── api_precision_compare_details_<timestamp>.csv

mindstudio_reports/<experiment-id>/<topology>/
precision_precheck/compare-r1/
├── precheck_report.html
└── precheck_index.json
```

`precheck_report.html` 是项目生成的可点击索引，逐 step/rank 链接官方 result、details
和 summary；它不重算阈值。`precheck_index.json` 保存相同关系和 fixture generation，
供 Release、自动审计或主报告发现。

阅读时先看 `api_precision_compare_result` 的 `Forward Test Success`、
`Backward Test Success` 和 `Message`，再进入 details 看具体 API。`pass` 表示该
API 在官方新精度标准下通过；`error` 才是待定位项；`SKIP` 可能表示 dtype、API、
黑白名单或运行条件不受支持，不能当作通过。官方新精度标准会按 API 选择绝对
阈值、标杆比较、二进制一致、ULP 或双千指标等判定，项目不二次伪造一套阈值。

预检能把问题缩到具体 API，但它仍是“从 capture 数据构造的单元测试”，不是完整
训练网络。最终结论仍要同时满足原始 dump compare 和多 step loss/grad norm；
对于依赖全局状态、通信、随机性或上下文的 API，还要回到完整模型复核。

## 10. 官方交付证据包

正式交付建议包含：

```text
A. 配置一致性
B. 单 step L0 statistics 官方比较
C. L1 API 预检及 GPU/NPU 预检结果比较
D. 可疑模块的 L1/tensor 原始数据官方比较
E. 代表性训练区间的 loss/grad norm
F. 两端重复运行稳定性
G. 可疑 API 的单算子复现与修复后 A/B
```

msProbe capture 只运行少量目标 step，因为它要保存模块/API 数据；长时状态监测只
保存关键统计。两者在信息密度和存储成本上互补，不应把 msProbe 扩成数千 step
全量 dump，也不应只看 loss 放弃模块定位。

## 11. 安全重跑与完整性

### 11.1 `--force`

`--force` 开始一个新的所选 generation：删除所选旧 run、artifact、report 后再
启动。数据阶段的 force 会同时重建 fixture，因此旧 capture 全部不能再使用。

### 11.2 不加 force

- 完整 artifact 且 experiment digest、fixture generation、附件 hash 一致：跳过；
- 目录存在但不完整：归档旧目录，重跑该成员；
- active PID：拒绝覆盖；
- compare 可重复生成，但只接受完整 artifact。

这保证一次中断后的续跑不会把上一次失败残片与本次新结果混搭。

### 11.3 工具改变必须改变证据

正式实验升级 msProbe revision 后，应锁定新 commit 并建立新 generation。即使
训练配置不变，工具 schema、支持 API、Result 判定或输出列也可能变化。

## 12. 已知限制

- msProbe 官方文档提示通用 PrecisionDebugger 对 PyTorch 2.7+ Dynamo 场景有
  限制；compile 必须使用专用 PrecisionChecker 流程；
- hook、`.item()`、CPU 同步和落盘可能改变 loss/gnorm 或性能；
- 原地操作及其相邻节点的反向数据可能因 autograd 机制缺失；
- 加速库可能检查函数对象类型，API wrapper 会触发兼容性问题；
- statistics 不能证明逐元素一致；
- tensor dump 可能包含训练数据、参数或梯度，属于敏感 raw；
- statistics 预检由数据特征构造输入，适合初筛但不等价于真实 tensor 预检；
- acc_check 不支持、被过滤或运行失败的 API 会产生 SKIP/error，需要阅读 Message；
- API 单测的 CPU 高精度标杆不能覆盖完整网络中的通信、全局状态和误差传播；
- 跨平台模块命名不同可能需要人工 mapping；
- 当前 adapter 没有自动编排所有 msProbe task，也没有替官方 GUI/CLI 重新实现。

## 13. 排障闭环

```text
配置不同
  → 修正配置，重建 fixture/capture

配置一致但 loss 异常
  → L0 statistics 找首个异常模块
  → L1/tensor 缩小到 API
  → stack 回到 TorchTitan/Turbo/torch_npu
  → 最小 A/B 修复
  → 重跑官方 compare
  → 重跑相同训练区间和官方比较

官方 compare 正常但 loss 长期漂移
  → 增加 dump step 或检查 optimizer/scheduler/data 顺序
  → 检查优化器、通信规约、离散 router/indexer 和累积误差
  → 增加目标 step 的 L0/L1 capture 或 Monitor 目标

精度正确但训练慢
  → 关闭 msProbe dump
  → 无采集运行建立基线
  → msProf / Ascend PyTorch Profiler / msprof-analyze / Insight 定位
```

这个顺序把“能跑、数值正确、长期稳定、性能高效”分成不同证据，避免用一个工具
回答它不负责的问题。
