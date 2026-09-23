# GLM-5.2 MindStudio 官方精度迁移流程

本流程是独立的 msProbe 实验合同。报告、判定、长程筛查和定位结论全部来自
本目录保存的官方工具输出与对应训练日志，不引用 parity 或其他 precision 实验。

标准流程的起点是正常训练，不是 dump。完成 CheckList 和可复现性准备后，先在相同
checkpoint、token plan、超参数和拓扑下分别运行 GPU/NPU **无 dump、无 Monitor
hook** 的正常训练，保存逐 step Loss、Grad Norm 和训练日志，再按现象选择定位工具。
官方稳定复现场景流程图在本项目中的顺序是：

```text
训练前配置检查       accuracy_benchmark.py --stage config-check
正常训练现象观察     accuracy_benchmark.py --stage baseline
  ├─ Loss/Grad Norm 是否出现 NaN/Inf？
  ├─ 第一步或前几步 Loss 是否不对齐？
  └─ 前期对齐后，长程 Loss/Grad Norm 是否漂移或尖刺？
按需长程状态监控     accuracy_benchmark.py --stage monitor
按需模块级快速定界   accuracy_benchmark.py --stage dump --level L0
按需 API 级下钻      accuracy_benchmark.py --stage dump --level L1/mix
按需 kernel 级下钻   accuracy_benchmark.py --stage dump --level L2
按需预检/溢出分析    --precheck / --overflow-check / nan_check
修复后闭环           重跑 baseline，再检查目标区间与最终任务指标
```

各类官方能力共用一个由固定实验合同生成的精度实验根，例如
`migration-cuda-npu-bf16-random-s2-b64-seq128-seed61-ffb9c634`。训练步数、
问题 step、dump task/level、Monitor 配置和诊断 case 只定义根目录下的操作 scope，
不能定义或替代整个实验。`--experiment` 仅用于把此前以人工别名保存的匹配数据迁入
规范实验根；新实验无需传入该参数。
`migration_benchmark.py`、
`configuration_check_benchmark.py`、`training_monitor_benchmark.py` 仅作为兼容入口保留。
已有人工别名目录会按字节不变的方式移动到新层级，完整采集仍可断点续跑，无需重采。

```text
mindstudio_{fixtures,runs,artifacts,reports}/accuracy/<experiment-id>/
└── <topology>/
    ├── inputs/<fixture-profile>/              # checkpoint、token plan、fixture
    ├── checklist/configuration-check/         # 训练前合同检查
    ├── observations/baseline/<training-profile>/ # 无工具 hook 的正常训练
    ├── captures/<dump-profile>/               # 任意 step/task/level 的 msProbe 采集
    └── observations/monitor/<monitor-profile>/# 任意长度的训练状态监测

mindstudio_artifacts/accuracy/<experiment-id>/diagnoses/<case-id>/
└── case.json                                  # 诊断控制面，不另建实验根
```

训练窗口没有固定的“几百步”或“5000 step”标准，必须覆盖已知问题或约定的验收
区间。baseline 始终先运行；只有正常训练表明需要更细的长期状态时才启用 Monitor
V2，只有确定可疑 step 后才用 L0 -> L1/mix -> tensor 或 L2 逐层下钻。禁止把
数百或数千 step 交给完整 dump。

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
纯 GPU 标杆端只安装基础 msProbe，不要求 CANN/torch_npu；GPU 数据不能离开内网
时的采集、单向汇合与本地 compare 配置见
[GPU 采集与内网离线比较环境](../toolchain/GPU_COLLECTION_AND_OFFLINE_ANALYSIS_ZH.md)。

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

## 2. 标准诊断流程与当前落地状态

msProbe 的五项工具能力不是整网诊断的起点。结合官方大模型训练精度定位流程，完整
顺序是：

```text
1. CheckList、固定随机性和确定性，确认问题可复现；
2. 正常 GPU/NPU 训练，观察 Loss、Grad Norm、NaN/Inf、尖刺和任务指标；
3. 先判 NaN/溢出，再判首 Step Loss，最后判长稳 Loss；
4. 根据现象选择 Monitor、模块/API dump、预检或溢出分析；
5. 修复后重跑局部证据、正常训练区间和最终任务指标。
```

当前仓库的落地边界如下：

| 阶段 | 目的 | 当前实现 |
| --- | --- | --- |
| 配置检查 | 找出两端 seed、dtype、优化器、模型、环境等差异 | 已接入 dynamic `ConfigChecker` 和逐 rank 官方 compare |
| 正常训练 | 在无 dump/Monitor hook 下记录逐 step Loss、Grad Norm 与所有数值日志指标 | 已由 `accuracy_benchmark.py --stage baseline` 接入并生成 JSONL/CSV/JSON/SVG |
| 训练状态监控 | 监控激活、梯度、权重、优化器及异常状态 | 已由 `accuracy_benchmark.py --stage monitor` 接入 Monitor V2 |
| 数据采集 | L0/L1/mix，statistics/tensor | 已由 `accuracy_benchmark.py --stage dump` 接入 `PrecisionDebugger` |
| 精度预检 | 对单端 API 构造单测、比较 CPU 高精度标杆，再比较 GPU/NPU 预检结论 | 已接入 `--precheck`、`--precheck-compare`，逐 step、逐 rank 保存官方结果 |
| 精度比对 | GPU/CPU golden 与 NPU target 比较 | 已由 `--compare` 调用官方 `msprobe compare` |

因此当前自动闭环是“固定输入 → 配置检查 → 正常训练现象分类 → 按需 Monitor →
按需 capture/API pre-check/official compare/分级可视化”。Monitor V2 只给
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

正式入口支持 `statistics`、`tensor`、`structure`、`overflow_check` 和
`nan_check`。`acc_check` 不作为 capture task：它由 `--precheck` 对已有 API
信息单独运行。`nan_check` 读取 NPU 寄存器状态，只允许 NPU L1，并由启动器设置
`INF_NAN_MODE_FORCE_DISABLE=1`；工具链源码构建必须包含 `nan_check` 模块。
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

跨服务器实验固定采用一条完整的数据流，不能把两端 capture 命令脱离同步步骤单独执行：

```text
NPU 生成唯一 fixture（checkpoint + token plan）
  -> NPU candidate capture
  -> GitHub Release full upload
  -> GPU full download
  -> GPU reference capture
  -> GPU 本地 compare / visualization
  -> GPU full upload，保存汇合后的完整实验
```

`full` 是继续实验所需的无损同步；`analysis` 仅供查看报告，不能用于另一端继续 capture。
后续 L0/L1/tensor、Monitor 和问题 step 下钻都复用这个规则：产生新 candidate 后从
NPU 上传，在 GPU 下载并完成 reference 与离线分析。Release 已存在时 upload 使用
`--clobber` 更新同一个规范实验，不创建第二个实验 ID。

### 5.1 生成固定实验输入

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --data --data-device npu --topology single
```

数据阶段生成并校验：

- 固定 token plan；
- 随机初始化 seed checkpoint；
- token/checkpoint hash；
- fixture generation ID；
- 生成命令和日志。

msProbe 不负责生成训练数据。它只观察使用这份契约的训练。

### 5.2 NPU candidate 与第一次同步

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture candidate --topology single

EXPERIMENT='migration-cuda-npu-bf16-random-s2-b64-seq128-seed61-ffb9c634'
python release_artifacts.py upload "$EXPERIMENT" --content full
```

### 5.3 GPU 恢复输入并采集 reference

```bash
EXPERIMENT='migration-cuda-npu-bf16-random-s2-b64-seq128-seed61-ffb9c634'
python release_artifacts.py download "$EXPERIMENT" \
  --backend wget --insecure --overwrite

export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture reference --topology single
```

下载得到的 NPU candidate 与本机生成的 GPU reference 位于同一规范实验根、同一
topology 和同一 capture profile。不能只复制 `dump.json`，否则会丢失 manifest、
fixture generation、工具版本和附件 hash。

### 5.4 compare

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --compare --topology single

python release_artifacts.py upload "$EXPERIMENT" --content full
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
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --compare --topology single --diff-analysis
```

真实 tensor compare 还可以输出单模块/API 日志或 XLSX：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --compare --topology single --tensor-log --xlsx
```

结构或名字无法自动配对时可传官方映射：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --compare --topology single \
  --data-mapping /absolute/path/data_mapping.yaml

python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
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
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump --list-topologies
```

一个分布式例子：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture reference --topology fsdp8

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture candidate --topology fsdp8

python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --compare --topology fsdp8
```

全拓扑：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture reference --topology all
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture candidate --topology all
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
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

### 7.0 当前阈值来源和判定规则

完整的变量定义、输入误差为零/非零分支、严格边界、数值例子和提示写入位置，
统一见[判定阅读指南 §3.1](MSPROBE_RESULT_READING_ZH.md#31-数值规则变量分支例子和标记位置)。
下方表格仅作索引，不应脱离上述定义理解“输入/输出误差”。

#### 本地源码审计更新（优先于下方文档规则摘要）

2026-09-07 实读 `D:/yyb/repos/msprobe`，HEAD 为
`86a64ee303cf27adedcefc0b661fd9d5ab9af615`，`git describe` 为
`tag_MindStudio_26.2.0.B050_002-8-g86a64ee3`。因此不能把这个 checkout
直接等同于服务器 wheel `26.1.0.post1`。以下是该 checkout 的确定实现：

源码统一前缀：`python/msprobe/core/compare/indicator_analysis/`。

| 判定 | 实际代码条件（严格不等号） | 文件/符号 |
| --- | --- | --- |
| statistics error | 最大输入 NormRelativeErr < 10%，且某输出 > 50%；标记该输出 | `algorithm.py::RelativeErrChecker` |
| statistics warning，输入非零 | 最大输出误差 / 最大输入误差 > 10；标记第一输出 | `algorithm.py::RelativeWarnChecker` |
| statistics warning，输入为零 | 最大输出 NormRelativeErr > 10%；标记第一输出 | 同上 |
| tensor error | 最小输入千分之一达标比例 > 90%，最小输出比例 < **10%**；标记第一输出 | `algorithm.py::OneThousandthErrChecker` |
| tensor warning | 最小输入 Cosine > 0.9，且最小输入减最小输出 > 0.1；标记第一输出 | `algorithm.py::CosineWarnChecker` |

`utils.py::str2float` 执行 `float(value.strip('%')) / 100`，因此统计阈值
0.1/0.5 明确是 10%/50%，不是 0.1%/0.5%。此前关于单位待核实的文字
仅适用于未审计的安装包，不再适用于该 checkout。

**源码与文字不一致：**`OneThousandthErrChecker.output_threshold = 0.1`，
但其 docstring、err_msg 和官方文档仍写 output < 0.6。该 checkout 的实际
执行以 0.1 为准，不能按错误提示解释成 60%。要确定服务器 wheel 行为，
需读取其同名类，不能仅根据发行版本号或本地 master 推断。

`TENSOR_CHECKERS` 未注册 Cosine > 0.99 或 MaxAbsErr < 0.001 的独立
硬阈值 checker；这两个数字是参考建议，不是自动 Result 通过线。
`STATISTICS_CHECKERS` 未对 Max/Min/Mean RelativeErr 设置独立数值阈值。
dtype、shape、requires_grad、标量、Inf/NaN 的 checker 另外参与判定。

compare 路径：`core/compare/acc_compare.py` →
`calculator.py::calculate_excel_result_df` → `ApiIndicatorCalculator`。
分级图路径：`visualization/builder/msprobe_adapter.py` →
`calculator.py::calculate_result` → 同一个 `ApiIndicatorCalculator`。
因此该源码中二者共享上述 checker；并行图合并有单独 checker 列表，
不应泛化到所有合并场景。当前单卡不涉及该例外。

核对日期：2026-09-07。项目的 `msprobe_adapter.compare_command` 和
`graph_visualize_command` 没有传入自定义误差阈值；报告也不会重新计算阈值。
实际判定来自执行命令时安装的 msProbe。当前服务器记录的版本为
`26.1.0.post1`，但本地尚未核验该安装包的判定源码；下面是官方当前文档
“比对结果（Result）”与“计算精度评价指标分析”的规则，不把 master 文档
冒充该 wheel 的源码审计。

官方依据：[精度比对文档](https://github.com/Ascend/msprobe/blob/master/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)。

| 模式/指标 | 官方 Result 规则或参考值 | 性质 |
| --- | --- | --- |
| statistics：Max/Min/Mean diff 及其 RelativeErr | 文档没有逐项统一数值 error 阈值 | 展示统计差异，不应自行补一条 BF16 容忍线 |
| statistics：NormRelativeErr | 输入 norm 相对误差 < 0.1 且输出 > 0.5，标记输出 error | 输入到输出误差扩大的联合规则，不是所有输出一律 > 0.5 就失败 |
| statistics：NormRelativeErr | 输出误差达到输入/参数误差的 10 倍，标记输出 warning | 零输入误差等边界需核对安装版本实现 |
| tensor：千分之一误差带内比例 | 输入/参数 > 0.9 且输出 < 0.6，标记输出 error | 比例是 90% 和 60%，误差带是逐元素 RE < 0.001 |
| tensor：Cosine | 输入/参数 > 0.9 且输入/参数减输出 > 0.1，标记输出 warning | 相似度下降规则 |
| tensor：Cosine | > 0.99 | 官方建议参考值，不等同于前一行自动 warning 规则 |
| tensor：MaxAbsErr | < 0.001 | 官方建议参考值，不是项目另行配置的硬阈值 |
| tensor：EucDist、MaxRelativeErr | 越接近 0 越好，文档未给统一通过线 | 不虚构默认阈值 |
| tensor：千分之五误差带内比例 | RE < 0.005 的元素比例 | 趋势指标，没有统一通过比例 |
| 两种模式：shape/dtype/requires_grad/非 tensor 标量 | 不一致可标记 error | 属于结构或属性问题，不能靠放宽数值阈值消除 |
| 两种模式：Max/Min 中 NaN/Inf | NPU 异常且标杆未出现相同现象时标记 error | 异常数值检查 |

**单位特别说明：**统计 RelativeErr 在表格中按百分数展示。官方规则文字写
`0.1`、`0.5`，不能未经安装包源码核对就声称这是 `0.1%/0.5%` 或
`10%/50%`；尤其 TensorBoard 的 graph_visualize 可能走不同的判定代码。
报告中的提示“greater than 0.5”不足以单独确定内部单位。确认后应在此记录
具体文件/函数、版本和转换方式，而不是凭界面猜测。

当前 `statistics + mix` 没有完整张量，不能计算真实逐元素 Cosine、MaxAbsErr
等 tensor 指标。统计 Norm 差为 `norm(N)-norm(B)`，不是 `norm(N-B)`。
所有摘要一致也不能证明元素一致，例如 `[1,2]` 和 `[2,1]`。

分级可视化是独立调用 `msprobe graph_visualize`，不是把 compare CSV 转成图。
模块边界 pass 不代表每个内部 API 都匹配；GPU/NPU 使用不同 RoPE 或融合实现时，
内部节点可以没有一对一关系。先核对语义、shape 和参数对应，再解释数值阈值。
工具运行成功、节点匹配成功和数值精度通过是三个不同结论。

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

完整命令遵循同一条 NPU -> Release -> GPU 汇合路径：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --data --data-device npu --topology single
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage config-check \
  --capture candidate --topology single

EXPERIMENT='migration-cuda-npu-bf16-random-s2-b64-seq128-seed61-ffb9c634'
python release_artifacts.py upload "$EXPERIMENT" --content full

# 以下在 GPU 服务器执行。
python release_artifacts.py download "$EXPERIMENT" \
  --backend wget --insecure --overwrite
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage config-check \
  --capture reference --topology single
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage config-check \
  --compare --topology single
python release_artifacts.py upload "$EXPERIMENT" --content full
```

分布式时不能只看 rank0。每个 rank 可能拥有不同 PP stage、DTensor shard 或环境，
所以项目逐 rank 生成包并逐 rank compare，再由总报告索引这些结果。

### 9.2 正常训练现象观察

完成 CheckList 后先执行 baseline，不安装 PrecisionDebugger 或 TrainerMonitorV2 hook：

```bash
# NPU candidate
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage baseline \
  --data --data-device npu --topology single --training-steps 500
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage baseline \
  --capture candidate --topology single --training-steps 500

# 上传 full，GPU 下载后执行 reference
export CUDA_VISIBLE_DEVICES=0
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage baseline \
  --capture reference --topology single --training-steps 500
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage baseline \
  --compare --topology single --training-steps 500
```

每端的 `training_metrics.jsonl` 记录 TorchTitan 正常 metrics logger 已产生的全部数值
指标；比较报告生成 `loss.svg`、`grad_norm.svg`、`relative_error.svg`、逐 step CSV 和
`summary.json`。摘要按官方流程先列出 NaN/Inf 的首次 step，再给出首个超过指导阈值
的 Loss step。它只做现象分类，不把 1% 指导值冒充所有模型的交付 PASS/FAIL。

观察顺序固定为：

1. Loss、Grad Norm 或其他训练指标是否出现 NaN/Inf；
2. 第一步是否不对齐；若第一步对齐，第二步或前几步何时首次不对齐；
3. 前几步对齐后，观察窗口内是否逐渐漂移或突然尖刺；
4. 若本窗口没有复现，扩大 baseline 窗口，而不是直接扩大 dump 窗口。

### 9.3 训练状态监控

官方训练状态监控面向长程运行中的激活、梯度、参数、优化器和通信异常。当前
`accuracy_benchmark.py --stage monitor` 在 Trainer 创建模型与优化器后调用
`TrainerMonitorV2.start()`，每个完整 train step 后调用一次 `step()`，并在正常
结束或异常退出时通过 `finally` 调用 `stop()`。官方没有固定训练步数；调用者必须
用 `--training-steps` 指定能够覆盖问题复现的窗口。默认只开官方案例常用的
weight_grad，module、optimizer、param 和 cc 根据现象显式开启。

这里必须区分 Monitor 和 dump：

```text
Monitor V2
  -> 选择一段长程 step 窗口
  -> 选择 gradient/module/param/optimizer/communication 等对象
  -> 记录 norm/mean/min/max/nans 等统计量
  -> 找到异常 step、rank、module 或 parameter

L0/L1/tensor dump
  -> 针对 Monitor 找到的窄范围
  -> 采集 Module/API 统计量或真实 Tensor
  -> 比较具体输入输出并定位首个异常计算
```

Monitor 不会因为选择 `weight_grad` 就保存完整梯度 Tensor。时间范围由
`--monitor-start-step`、`--monitor-stop-step` 和 `--monitor-step-interval` 控制；
其中 start inclusive、stop exclusive，`--training-steps` 仍需覆盖 stop。对象和用途为：

| 开关 | 监测对象 | 典型用途 |
| --- | --- | --- |
| 默认 `weight_grad` | 参数梯度 | Grad Norm、梯度尖刺、NaN/Inf、异常参数 |
| `--monitor-module` | 模块激活/梯度统计 | Loss 先异常或需要缩小异常模块 |
| `--monitor-param` | 参数统计 | 权重逐步漂移或参数更新异常 |
| `--monitor-optimizer` | Adam 等优化器状态 | 一阶/二阶矩或融合优化器定界 |
| `--monitor-cc` | 通信相关统计 | rank 或通信阶段差异筛查 |

`weight_grad` 会产生两个 `scope`：

- `unreduced`：在反向传播阶段通过梯度 hook 记录，更接近梯度生成与累积过程；
- `reduced`：在调用 `optimizer.step()` 前记录，是当前 step 最终交给优化器前的梯度形态。

这两个名字不能脱离并行实现机械理解成“一次 AllReduce 的严格前后”。在普通
DDP、FSDP、梯度累积、gradient clipping 或其他梯度变换下，两个采集点之间可能包含
不同处理。官方 Monitor V2 会在 PyTorch FSDP 场景自动尝试 reduce 前采集
`unreduced`；项目仍需在目标拓扑检查参数覆盖、scope 和数值是否符合实际执行顺序。

官方定义、配置字段和 CSV schema 见
[msProbe Monitor V2 使用指南](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/monitor_v2_instruct.md)。当前框架对应配置生成见
`tests/glm5_2_mindstudio/config.py::MsProbeMonitorConfig`，Trainer 生命周期接入见
`tests/glm5_2_mindstudio/capture_training.py::_install_training_monitor`。

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
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage monitor \
  --data --data-device cuda --topology single
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage monitor \
  --capture reference --topology single --training-steps 5000

unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage monitor \
  --capture candidate --topology single --training-steps 5000

python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage monitor \
  --compare --topology single --training-steps 5000
```

已知异常大约发生在 step 360，并且 Grad Norm 先于 Loss 上扬时，应先只采
weight gradient 的统计量。例如在 NPU FSDP8 上覆盖 step 300--400：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage monitor \
  --capture candidate --topology fsdp8 \
  --training-steps 401 \
  --monitor-start-step 300 --monitor-stop-step 401 \
  --monitor-step-interval 1 \
  --monitor-ops norm,mean,min,max,nans
```

分析时先按 step 对齐 Loss、Grad Norm 与 Monitor CSV，再按 rank、parameter 和
`scope=unreduced/reduced` 找到最早异常。如果 `unreduced` 已异常，优先下钻反向
生成和累积链路；如果 `unreduced` 正常而 `reduced` 异常，则检查两采集点之间的
通信、累积、裁剪及其他梯度处理。确定第一现场以后，再对该 step 的 backward 或
“上一步 backward + 当前 forward”运行窄范围 dump。这里得到的是诊断方向，不是
Monitor 自动给出的根因或跨设备 PASS/FAIL。

每个端点的 `official/rank_<rank>/**/*.csv` 是权威监控数据；compare 只生成
`official_compare/monitor_index.json` 将两端身份关联起来。官方 Monitor V2 没有
GPU/NPU cross-device 数值 comparator，所以这里的 `unparsed`/索引完成不是 FAIL，
也不能改写成 PASS。先从 CSV 找出异常 step/rank/module，再在对应 step 运行 L0/L1
dump。Monitor 的 CSV、训练日志和官方 dump/compare 共同组成这套独立 msProbe
实验的长程与定位证据，不引用其他实验目录的结论。

分布式先验证 FSDP/TP/EP 的 sharded 参数与 optimizer 容器，再验证 PP 的 model-parts
名称和 optimizer ownership。当前 workflow 会警告 PP 尚需服务器专项验收，并拒绝
Monitor V2 与正式 `--compile.enable` 组合，避免静默产生不可解释数据。

### 9.4 分级图可视化

分级图不是重新 capture。它读取同 generation、同 L0/mix 配置的 GPU/NPU
`construct.json`，调用官方 `graph_visualize` 生成 `.vis.db`：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
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

### 9.5 precision pre-check

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
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --data --data-device cuda --topology single \
  --level L1 --dump-task statistics
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture reference --topology single \
  --level L1 --dump-task statistics

export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture candidate --topology single \
  --level L1 --dump-task statistics
```

再在两端分别执行 checker：

```bash
# GPU server
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --precheck reference --topology single \
  --level L1 --dump-task statistics

# NPU server
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
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
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
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
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
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
启动器同时管理完整进程组：`Ctrl+C`、`SIGTERM` 或 SSH/tmux 断开产生的
`SIGHUP` 会先停止并回收当前 topology、torchrun 和全部 rank，再把本次状态标记为
`interrupted` 并释放锁。不可捕获的 `SIGKILL` 或主机掉电只能留下 dead-owner
状态；下一次不加 force 的运行会把该成员视为未完成并重跑，不能将其当作完成结果。

### 11.3 工具改变必须改变证据

正式实验升级 msProbe revision 后，若要得到新工具版本的证据，应锁定新 commit，
并显式使用 `--force` 建立新 generation（或使用新的实验 identity）。即使训练配置
不变，工具 schema、支持 API、Result 判定或输出列也可能变化。当前工具版本和源码
身份只作为旧 capture 的 provenance 保存；它们发生变化时，不加 `--force` 不会把
已经完成的历史 capture 静默判成失败并重新采集。

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
- 框架无关的官方 task 已由统一 capture CLI 编排；Megatron/verl 专用能力不冒充
  TorchTitan 能力。

## 趋势可视化与单边构图

L0/mix dump 完成后，可直接生成单边结构图或双边比较图：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --graph-visualize --graph-side candidate --topology single --level L0

python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --graph-visualize --graph-side compare --topology single --level L0 \
  --layer-mapping mapping.yaml
```

多 step、多 rank 或 Monitor V2 数据使用官方 `msprobe data2db` 转成趋势数据库：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --trend candidate --topology all --level L0 --trend-format dump

python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage monitor \
  --trend candidate --topology all --training-steps 5000 \
  --trend-format monitor --trend-processes 8
```

每个成员生成独立 `.trend.db`，终端同时打印数据库路径与 TensorBoard 命令。
TensorBoard 中选择 `TREND ANALYZER`，可沿 Step、Rank、Module Name 三个维度查看
热力图和折线趋势。

官方 graph merge 与 `compare --consistent_check` 是框架特定能力：前者当前按
Megatron rank order 合并且不支持 CP，后者只用于 verl 的受支持模型。adapter 保留
官方命令构造能力，但不向 GLM5 通用 CLI 暴露，防止把参数可执行误报为 TorchTitan
语义已适配。

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

## 14. kernel 级与首个溢出节点下钻

只有 L1 已把异常缩到 NPU 算子、仍需观察 kernel 级输入输出时，才建立独立 L2
generation。L2 数据不能和原有 L0/L1 artifact 混用：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --data --data-device npu --topology single \
  --level L2 --dump-steps 0,1 --force

python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --capture candidate --topology single \
  --level L2 --dump-steps 0,1
```

当 dump 或 Monitor 发现 INF/NaN 时，对已经完成的官方 capture 执行首节点分析：

```bash
python tests/glm5_2_mindstudio/accuracy_benchmark.py --stage dump \
  --overflow-check candidate --topology single \
  --level L0 --dump-steps 0,1
```

输出位于当前实验 report 的 `overflow-candidate-r1/stepN/`，其中
`runtime.log`、`invocation.json` 和 msProbe 原始输出共同构成证据。该命令调用
官方 `msprobe overflow_check`，项目不重写其传播关系或结论。
