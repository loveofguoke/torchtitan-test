# msProbe：PyTorch 编译精度比对

## 用途

`PrecisionChecker` 用来逐模块比较 PyTorch eager 与 `torch.compile` 的前向输出和梯度。当 eager 正常，而 compile 后出现 loss 抖动、收敛变差或输出不一致时，用它定位是哪个编译单元、前向还是反向首先产生差异。

官方文档：[PyTorch 场景编译精度比对](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md)

## 模式选择

| 场景 | 模式 | 调用方式 | 关键限制 |
| --- | --- | --- | --- |
| 训练，包含 backward | single-pass，默认 | `install()` + 原训练 step + `collect(loss)` | 不提供真实 eager 整网 loss |
| FSDP2 多卡训练 | single-pass，必须 | 每 rank 分别 `install/collect/report` | 工具不内置多卡 CSV 汇总 |
| 推理/eval，仅 forward | two-pass，推荐 | `compare(run_step, model)` | 会 `deepcopy` 模型 |
| 需要真实 eager/compiled 整网 loss | two-pass | `compare(run_step, model)` | 不支持 FSDP2 或其他禁止 `deepcopy` 的模型 |

single-pass 在 compiled 子模块的 forward hook 中，用相同输入重新执行未编译的 `_orig_mod`，然后比较两侧模块输出；它不复制模型，也不另跑完整 eager 训练链路。报告中的 `loss_eager=NaN` 是预期行为，不能据此判失败。

two-pass 构造 eager、compiled 两份模型并分别执行同一个 `run_step`，可以比较真实的 `loss_eager`、`loss_compiled`、模块输入输出和梯度。

## 当前版本的判定阈值

比较前，参与比较的 Tensor 会转成 CPU FP32。逐 Tensor 指标为：

| 指标 | 公式 |
| --- | --- |
| `max_abs` | `abs(a-b).max()` |
| `mean_abs` | `abs(a-b).mean()` |
| `max_rel` | `(abs(a-b) / (abs(a)+1e-8)).max()` |
| `allclose` | `torch.allclose(a, b, atol=1e-4, rtol=1e-3)` |
| `shape` | shape 不一致直接失败 |

最重要的版本细节：`PrecisionChecker(threshold=...)` 的默认值虽然是 `1e-4`，但官方文档明确说明该参数当前是预留参数；实际 PASS/FAIL 固定使用 `atol=1e-4, rtol=1e-3`。在本次核对的官方实现中也确实是硬编码的 `torch.allclose`。

因此：

- 不要认为修改 `threshold` 已改变判定门槛。
- 如果本项目需要 BF16/FP32 分 dtype 门槛，应读取 CSV 的误差字段，用项目自己的分析/验收逻辑二次判定。
- `max_rel` 分母使用 eager 侧 `abs(a)+1e-8`，接近 0 的元素可能使最大相对误差很大；PASS/FAIL 仍以 `allclose` 为准。

## 训练场景标准接入

```python
import torch.nn as nn
from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker

checker = PrecisionChecker()                 # single_pass=True
checker.wrap_by_policy(model, (nn.Linear,))  # 选择编译和检查单元
checker.install(model)                       # 必须在训练 step 前

loss = run_step(model)                       # forward -> loss -> backward
result = checker.collect(loss)               # 必须在该 step 后
checker.report(result, csv_path="precision_report.csv")
```

支持的范围选择方式：

- `wrap(module)`：指定单个模块。
- `wrap_by_policy(model, (ModuleType, ...))`：按类型选择。
- `wrap_all_children(model, depth=...)`：按层级递归选择，自动跳过 `ModuleList`、`ModuleDict`、`Sequential` 等容器。
- `ignore(module)` / `ignore_by_policy(...)`：模块仍执行，但不参与报告判定。

大型模型应从粗粒度开始，例如先 wrap transformer block；找到失败 block 后再拆到 attention、MLP，最后拆到更小的线性层或算子封装。若父模块已 wrap，其内部子模块不会再作为独立顶层编译目标。

## two-pass 示例

```python
from msprobe.pytorch.compile_accuracy_checker.precision_checker import PrecisionChecker

checker = PrecisionChecker(single_pass=False)
checker.wrap_all_children(model)

def run_step(model):
    model.zero_grad()
    output = model(inputs)
    loss = loss_fn(output, labels)
    loss.backward()
    return loss

result = checker.compare(run_step, model)
checker.report(result, csv_path="precision_report.csv")
```

推理没有 loss 时，可以返回 `logits.mean()` 等标量作为对比锚点；它只用于给出整网 eager/compiled 数值，不需要 backward。

## 报告解读

stdout/CSV 将检查项分为：

- `FORWARD INPUT`：开启 `capture_input` 时的模块输入，用于区分输入传播差异与模块自身引入差异。
- `FORWARD OUTPUT`：模块输出。
- `BACKWARD`：two-pass 比较 `grad_input`、`grad_output`；single-pass 当前主要尝试重建 `grad_input`，`grad_output` 通常为 `None`。
- `LOSS`：two-pass 有双侧真实值；single-pass 只有 compiled 值。

CSV 字段包括 `module_name`、`check_type`、`tensor_index`、`status`、`max_abs_diff`、`mean_abs_diff`、`max_rel_diff`、`shape`、`note`。

常见状态：

| 状态/说明 | 含义 |
| --- | --- |
| `PASS` | shape 一致且 `allclose` 通过 |
| `FAIL` | shape 不同或 `allclose` 不通过 |
| `IGNORED` | 被显式忽略 |
| `SKIP_compiled_wrapper` | compiled wrapper 的部分 backward hook 不直接对齐，通常是预期跳过 |
| `SKIP_inside_compiled` | 模块位于已编译模块内部，融合后 hook 未触发 |
| `MISSING_fwd/bwd_*` | 某一侧未采集到，需要排查 wrap 粒度或 hook 是否触发 |

分析时优先找：`FORWARD INPUT` 仍 PASS，但 `FORWARD OUTPUT` 首次 FAIL 的模块；若前向都通过而 `grad_input` 首次 FAIL，则重点检查反向编译图。

## FSDP2 多卡注意事项

FSDP2 参数分片后禁止 `deepcopy`，所以只能使用 single-pass。每个 rank 独立生成报告：

```python
rank = torch.distributed.get_rank()

checker = PrecisionChecker()
checker.wrap_by_policy(model, (TransformerBlock,))
checker.install(model)

loss = run_step(model)
result = checker.collect(loss)
checker.report(result, csv_path=f"precision_rank{rank}.csv")
```

官方工具当前不自动汇总多卡 CSV。项目汇总时应在原有字段前增加 `rank` 和 `scenario`，并保留每个 rank 的原始报告，避免把只发生在单 rank 的差异平均掉。

## 其他能力与限制

- `cast_dtype=torch.bfloat16` 等配置会为被 wrap 模块启用 autocast，可检查混合精度编译差异。
- `dump_graphs=True` 可保存 Dynamo 捕获的 graph，但依赖 PyTorch 内部接口，版本变化可能使行为改变。
- `install()` 会原地替换被 wrap 的子模块；若要检查根模型，优先使用 `compare()`，或改为 wrap 根模型下的子模块。
- 随机算子、状态更新和输入不确定性会污染 eager/compile 对比。诊断用例应固定随机性，并记录模型模式和 autocast 配置。

## 本项目建议

对于 GLM5 + FSDP2 的编译精度问题：

1. 使用单个确定性 step，固定输入 fixture 和 seed。
2. 必须用 single-pass，每 rank 输出单独 CSV。
3. 先 wrap block；再拆 attention、indexer/router、MLP 等候选模块。
4. 保存 `FORWARD INPUT/OUTPUT` 与 `grad_input`，从首个失败边界定位。
5. 需要自定义 BF16 门槛时基于 CSV 二次判断，不依赖当前无效的 `threshold` 构造参数。
