# NPU Inductor 算子后端选择

公共参数 `--npu-codegen dvm|ascend-triton` 与图模式独立。
省略保持原策略；显式选择写入子进程环境和实验配置，不修改模型数学。
A2/A3 实验选 dvm，A5 实验选 ascend-triton；不自动猜测硬件。

| CLI | TorchNPU 环境变量 TORCHINDUCTOR_NPU_BACKEND |
|---|---|
| dvm | dvm |
| ascend-triton | default |

`default` 是已核对的 TorchNPU Triton loader 名，不传 `backend="dvm"`。
整模型编译仍使用 Inductor。eager 模型内部的 FlexAttention 也可能局部编译，
所以 eager 同样允许指定 codegen。该参数不启用 eager lazy fusion。

## 命令

```bash
# Smoke，先单卡后全部拓扑；结果目录包含 dvm 标签。
python tests/glm5_2_smoke/train_smoke.py --device npu --topology single --graph inductor --npu-codegen dvm --steps 2
python tests/glm5_2_smoke/train_smoke.py --device npu --topology all --graph inductor --npu-codegen dvm --steps 2

# MindStudio 性能：后端选择不隐式开启整模型编译。
python tests/glm5_2_mindstudio/performance_benchmark.py --probe --device npu --topology single --preset distributed --npu-codegen dvm --extra-train-arg=--compile.enable --extra-train-arg=--compile.backend=inductor
```

同一参数也接入原 performance workflow、组合实验、checkpoint、原 precision
topology suite 和 MindStudio accuracy CLI。已有图模式参数保持原义：
组合实验仍由 reference/candidate graph 决定两侧执行模式；新 codegen 参数应用于两侧。
MindStudio/precision 的跨设备实验只向 NPU endpoint 注入，不向 GPU 注入。
需要分别指定两侧不同 NPU 后端时，当前请使用各自 endpoint.environment 配置。

显式选择会改变配置/运行标识，不应复用另一后端的完成结果。
首次开始新后端无需 force；同一配置中断后不加 force 续跑。
不要依靠未记录的外部环境变量切换后端。缓存建议按后端单独设置：

```bash
export TORCHINDUCTOR_CACHE_DIR=/workspace/y50064852_yyb/temp/torchinductor_dvm
unset TORCH_NPU_LAZY_FUSION
```

## 已验证环境与边界

910B2、CANN 9.1.0、torch 2.14.0.dev20260805+cpu、torch_npu 2.14.0：
独立 FP32 pointwise demo 的前向/反向数值检查通过，生成代码包含
`@dvm.kernel(ktype='vector', dyn_shape=False)` 和 `.run(...)`。
这是服务器已有实验，不是本次框架测试结果。该用例未安装 torch_mlir 仍可运行。
不代表 BF16 GLM、DSA FlexAttention 或分布式已通过；编译错误应保留原始日志，不静默回退。

参考：[DVM TorchNPU 接入](https://gitcode.com/mindspore/dvm/blob/master/docs/pytorch.md)、
[TorchNPU loader](https://gitcode.com/Ascend/pytorch/blob/master/torch_npu/_inductor/__init__.py)。
