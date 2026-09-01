# msProbe：PyTorch 离线精度预检

## 用途

离线精度预检把 NPU 训练中采集到的 L1/API 输入构造成单元测试，分别得到 NPU 对 CPU 高精度标杆、GPU 对 CPU 高精度标杆的结果，再比较两套结果，以判断 NPU API 的误差是否明显差于 GPU。

官方文档：[PyTorch 场景离线精度预检](https://www.hiascend.com/document/detail/zh/mindstudio/2610/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_checker/pytorch_accuracy_checker_instruct.md)

它与整网 `msprobe compare` 的区别：整网比对关注一次真实执行链路中的首差异；预检关注由输入构造出的单 API 正确性，更适合快速筛出可疑 API。

## 操作流程

1. 在 NPU 与 GPU 环境安装相同版本的 msProbe。
2. 在 NPU 训练脚本中用 `PrecisionDebugger` 采集预检数据，`level` 必须为 `L1`。
3. 把 NPU dump 数据复制到 GPU 环境。
4. 在 NPU、GPU 环境分别执行 `acc_check`；小规模单卡用 `acc_check`，大模型数据量大时用 `multi_acc_check`。
5. 把两侧 `accuracy_checking_details_*.csv` 放到同一环境。
6. 执行 `api_precision_compare`，以其输出作为最终预检结果。

单侧预检示例：

```bash
msprobe acc_check \
  -api_info ./dump_path/step0/rank0/dump.json \
  -o ./acc_check_result
```

需要保存未达标输入输出时添加 `-save_error_data`。`-j`/`--jit_compile` 用于开启 JIT 编译；继续中断任务时用 `-csv_path` 指向上次的 `accuracy_checking_result_*.csv`。

双侧结果比较：

```bash
msprobe api_precision_compare \
  -npu ./npu/accuracy_checking_details_<timestamp>.csv \
  -gpu ./gpu/accuracy_checking_details_<timestamp>.csv \
  -o ./api_precision_compare
```

先看 API 级 `*_result_*.csv` 的前向/反向状态，再按 API 名进入 `*_details_*.csv` 查看每个输出的指标。一个 API 有多个输出时，必须全部 pass 才整体 pass；只要一个输出 error，整体为 error；只有 warning/pass 且无 error 时整体为 warning。

## 初步 `acc_check` 指标与门槛

初步结果依次使用以下规则：

1. `Cosine > 0.99`；否则直接 error。
2. `MaxAbsErr < 0.001`；满足则 pass，不满足再看分 dtype 的元素比例指标。
3. FP16/BF16：双百不通过为 error；双百通过而双千不通过为 warning；双百与双千均通过为 pass。
4. FP32/FP64：双千不通过为 error；双千通过而双万不通过为 warning；双千与双万均通过为 pass。

元素比例指标定义：

| 名称 | 含义 | 通过条件 |
| --- | --- | --- |
| 双百 | 相对误差 `< 1%` 的元素比例 | 相对误差 `> 1%` 的元素比例 `< 1%` |
| 双千 | 相对误差 `< 0.1%` 的元素比例 | 相对误差 `> 0.1%` 的元素比例 `< 0.1%`，即指标 `> 0.999` |
| 双万 | 相对误差 `< 0.01%` 的元素比例 | 相对误差 `> 0.01%` 的元素比例 `< 0.01%` |

初步 `accuracy_checking_result_*.csv` 是中间参考结果；官方建议继续执行 NPU/GPU 的 `api_precision_compare`。

## 小值域规则

小值使用绝对误差，正常值使用相对误差。官方阈值如下：

| dtype | 判定为小值：`abs(bench)` 小于 | 小值绝对误差阈值 |
| --- | --- | --- |
| `torch.float32` | `2**-14` | `2**-30` |
| `torch.float16` | `2**-11` | `2**-16` |
| `torch.bfloat16` | `2**-8` | `2**-16` |

这样可以避免标杆接近 0 时相对误差无限放大。

## `api_precision_compare` 的综合判定

最终比较可能采用标杆比对法、绝对阈值法、二进制一致法、ULP 误差法或双千指标法。算子看护等级越高，标杆比对容忍度越严格。

### 标杆比对法

比值通常表示“`NPU vs CPU` 误差 / `GPU vs CPU` 误差”。

| 指标 | L0 | L1 | L2 |
| --- | ---: | ---: | ---: |
| 小值域错误比值 | `<= 2` | `<= 2` | `<= 2` |
| 均方根误差比值 | `<= 2` | `<= 1.5` | `<= 1.2` |
| 相对误差最大值比值 | `<= 10` | `<= 5` | `<= 2` |
| 相对误差平均值比值 | `<= 2` | `<= 1.5` | `<= 1.2` |

误差均衡性比值 `<= 2` 为 pass，但当前不参与 API 汇总结果判定。

未列入 L1/L2 的算子默认按 L0 看护。官方 L1 示例包括 `embedding`、`npu_fusion_attention` 及其梯度、`scatter_`、`scatter_add_`；L2 示例包括 `matmul`、`mm`、`addmm`、`layer_norm`、`GELU`、`sum`、`npu_rms_norm` 等。具体列表可能随版本变化，必须查当前官方文档。

### 绝对阈值与二进制一致

- inf/NaN 错误率必须为 0。
- 正常值的相对误差错误率必须为 0。
- 小值的绝对误差错误率必须为 0。
- bool、整数、部分 builtin 或指定 API 的二进制一致错误率必须为 0。
- 配置为量化 API 且“浮点输入、整数输出”时，官方配置允许绝对误差 `<= 1` 判为 pass。

### ULP 误差

| dtype | “ULP 超阈值”中的 ULP 阈值 | 通过条件（满足任一） |
| --- | ---: | --- |
| FP16/BF16 | `1` | NPU 超阈值比例 `< 0.001`；或 NPU 比例小于 GPU 比例 |
| FP32 | `32` | NPU 平均 ULP `< 64`；或超阈值比例 `< 0.05`；或 NPU 比例小于 GPU 比例 |

### 双千指标法

当前官方说明该方法仅用于 `conv1d`、`conv2d`；双千指标 `> 0.999` 为 pass，否则 error。

## 解释结果时的注意点

- `SKIP` 不代表 pass：可能是 dtype/反向不支持、黑白名单过滤、无计算、运行错误或 API 未被支持。
- `warning` 也不应计为严格通过；需要结合详细指标和本项目标准复核。
- CPU 高精度不是 GPU 结果本身；最终比值衡量的是 NPU 相对 CPU 的误差是否明显差于 GPU 相对 CPU 的误差。
- 随机生成模式快但只能粗筛；需要准确判断时使用真实数据模式。
