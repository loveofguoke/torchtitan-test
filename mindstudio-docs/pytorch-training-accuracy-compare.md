# msProbe：PyTorch 训练精度比对

## 目标与适用范围

该能力比较 NPU 侧与 CPU/GPU 标杆侧在相同模型执行过程中的 Module/API 前向、反向输入输出，用于定位迁移、框架升级或硬件升级后出现的精度差异。

官方文档：

- [PyTorch 场景精度数据采集](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/dump/pytorch_data_dump_instruct.md)
- [PyTorch 场景精度比对](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [PyTorch 场景快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/quick_start/pytorch_quick_start.md)

## 先保证比较有效

两侧至少需要对齐：

- 相同代码、模型结构、checkpoint、输入数据及 batch 顺序；
- 相同随机种子、采集 step、rank 对应关系；
- 相同的训练/eval 状态、优化器状态、梯度清零位置；
- 明确记录两侧 dtype、autocast、loss scaling、确定性配置和并行切分方式。

msProbe 使用 `PrecisionDebugger` 的 hook 和部分同步操作，官方说明工具接入后可能影响 loss/gnorm；因此应把采集运行视为诊断运行，不直接拿其吞吐或训练时序作为无工具运行的性能结论。

## 采集粒度与模式

### 粒度

| `level` | 内容 | 建议用途 |
| --- | --- | --- |
| `L0` | Module 级 | 先定位到 block、attention、MLP 等大模块 |
| `L1` | API 级 | 对可疑模块继续定位到具体 PyTorch API |
| `mix` | L0 + L1 | 需要同时还原层级关系和 API 细节时使用，数据更多 |

### 数据模式

| 配置 | 采集内容 | 可用结果 | 建议用途 |
| --- | --- | --- | --- |
| `task: "statistics"`、`summary_mode: "statistics"` | Max、Min、Mean、L2 Norm 等统计量 | 统计量绝对/相对误差 | 低开销粗定位 |
| `task: "tensor"` | 统计量和完整 Tensor | Cosine、欧氏距离、最大绝对/相对误差、元素比例 | 可疑点深挖 |
| `task: "statistics"`、`summary_mode: "md5"` | 统计量和 CRC-32 | 一致性检查 | 确定性问题快速筛查 |
| `task: "statistics"`、`summary_mode: "xor"` | XOR 校验值 | 轻量一致性检查 | 更低开销的确定性筛查 |

## 标准操作过程

### 1. 两侧使用一致配置采集

示例配置用于采集第 0、1 step 的 Module/API 统计量：

```json
{
  "task": "statistics",
  "dump_path": "/path/to/msprobe_dump",
  "rank": [],
  "step": [0, 1],
  "level": "mix",
  "async_dump": false,
  "statistics": {
    "scope": [],
    "list": [],
    "data_mode": ["all"],
    "summary_mode": "statistics"
  }
}
```

训练代码的核心接入顺序：

```python
from msprobe.pytorch import PrecisionDebugger, seed_all

seed_all()
debugger = PrecisionDebugger(config_path="./config.json")

for batch in data_loader:
    debugger.start(model)
    loss = train_step(model, batch)
    debugger.stop()
    debugger.step()
```

`start()` 开始采集，`stop()` 停止当前采集区间，`step()` 完成本 step 落盘并推进采集 step。GPU 与 NPU 应采集同一个逻辑 step 和同一份输入。

### 2. 执行比对

单卡：

```bash
msprobe compare \
  -tp /path/to/npu/step0/rank0/dump.json \
  -gp /path/to/gpu/step0/rank0/dump.json \
  -o ./accuracy_compare
```

多卡时，`-tp` 和 `-gp` 指向 rank 上一级的 step 目录：

```bash
msprobe compare \
  -tp /path/to/npu/step0 \
  -gp /path/to/gpu/step0 \
  -o ./accuracy_compare
```

常用选项：

- `--xlsx`：输出 xlsx；默认输出 csv。
- `-fm`：同一层级、同名但调用次数不同的 API 做模糊匹配。
- `-dm mapping.yaml`：API/Module 无法自动匹配时提供显式映射；该场景不支持 `-fm`。
- `-cm cell_mapping.yaml`：比较不同平台或不同配置下的 Module。
- `-da`：分析首差异节点，输出逐 rank 结果和 `diff_analyze_*.json`。
- `-tensor_log`：打印单个 Module/API 的 Tensor 比对日志，仅适用于 tensor 数据。

### 3. 按顺序阅读结果

1. 先筛选 `Result=error/warning`。
2. 查看 `Err_Message`，先排除 shape、dtype、`requires_grad`、标量参数、匹配关系或非有限值问题。
3. 确认异常节点的输入是否已经偏离；输入已偏离时继续向前找，输入正常但输出显著恶化时才优先怀疑当前节点。
4. 使用 `NPU_Stack_Info` 或 NPU dump 中的 `stack.json` 回溯训练代码。
5. 粗定位完成后缩小 `scope/list`，切换到 `tensor` 复核真实数据。

API 自动匹配通常要求名称相同、输入输出 Tensor 数量相同。名称包含 API 类型、API 名、调用次数、前后向、输入输出和索引；两侧控制流或调用次数不同会造成错配，必须先解决或显式映射。

## 指标与公式

记 NPU Tensor 为 `N`，标杆 Tensor 为 `B`，逐元素相对误差为 `RE = |(N-B)/B|`。

### 真实 Tensor 模式

| 指标 | 定义/解释 | 官方参考值 |
| --- | --- | --- |
| `Cosine` | `dot(N,B) / (||N||₂ ||B||₂)` | 可接受值 `> 0.99`；零向量可能产生 NaN |
| `EucDist` | `||N-B||₂` | 越接近 0 越好，无统一通过阈值 |
| `MaxAbsErr` | `max(|N-B|)` | 可接受值 `< 0.001` |
| `MaxRelativeErr` | `max(RE)` | 越接近 0 越好；标杆含 0/NaN 时出现 inf/NaN 不一定异常 |
| `One Thousandth Err Ratio` | `count(RE < 0.001) / numel` | 趋势指标，不直接参与精度通过判定 |
| `Five Thousandth Err Ratio` | `count(RE < 0.005) / numel` | 趋势指标，不直接参与精度通过判定 |

`Cosine > 0.99` 和 `MaxAbsErr < 0.001`是官方给出的通用参考，不应脱离 dtype、数值尺度和业务容忍度单独作为本项目最终验收门槛。

### 统计量模式

采集 Max、Min、Mean、L2 Norm，并输出两侧统计量差值及相对误差。例如：

- `Norm diff = L2(N) - L2(B)`；
- `NormRelativeErr = |(L2(N)-L2(B))/L2(B)| × 100%`。

统计量模式适合发现异常放大、分布漂移和首差异范围，但不能替代完整 Tensor 比对。

### MD5/XOR 模式

用于检查数据是否严格一致。校验值不一致只能说明数据不同，不能给出误差大小，也不能自动区分可接受的浮点舍入与真实精度缺陷。

## `Result` 的官方启发式规则

`Result` 有 `pass`、`warning`、`error`，优先级为 `error > warning > pass`。它不是简单地把上述每个参考阈值做 AND。

重要规则包括：

- 真实数据模式：输入/参数的千分之一误差比例 `> 0.9`，但输出 `< 0.6`，输出标记为 error。
- 统计量模式：输入 Norm 相对误差 `< 0.1`，但输出 Norm 相对误差 `> 0.5`，输出标记为 error。
- 统计量模式：输出 Norm 相对误差达到输入/参数的 10 倍，输出标记为 warning。
- 真实数据模式：输入/参数 Cosine `> 0.9`，但输出相对输入的 Cosine 劣化超过 `0.1`，输出标记为 warning。
- shape、dtype、`requires_grad` 或非 Tensor 标量参数不一致会标记为 error。
- NPU 出现 NaN/Inf/-Inf 而标杆没有同类现象会标记为 error。
- MD5 模式下 CRC-32 不一致为 error；参数未匹配为 warning。

涉及输入传播关系的规则不适用于部分带占位输入的通信 API。`empty`、`empty_like`、`to` 等冗余/分配类 API 也不适合按常规指标解释。

## 本项目落地建议

- 首轮使用 `statistics + mix + 少量 step`，避免直接产生巨量 Tensor 文件。
- 发现某一 GLM block 首先劣化后，改用 L0 缩小到 attention/MLP，再用 L1 + tensor 定位 API。
- 多 rank 问题要保留每个 rank 的结果；通信前输入一致、通信后首次分叉时再重点检查 rank 映射和通信路径。
- 报告中后续大量节点异常通常是误差传播结果，优先处理“输入仍对齐、输出首次显著不对齐”的节点。

## 已知限制

- NPU 自研 API 在标杆侧没有对应 API 时不会自动比较。
- 原地操作及其相邻模块可能缺少反向 dump。
- 两侧 API 调用次数或控制流不同可能导致无法匹配或错配。
- 大规模 tensor dump 对磁盘、I/O 和比较时间开销很大，应先缩小范围。
