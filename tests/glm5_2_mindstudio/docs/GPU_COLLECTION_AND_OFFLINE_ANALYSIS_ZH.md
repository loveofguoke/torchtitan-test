# GPU 采集与内网离线对比环境

本文只描述 MindStudio 标准流程中的 GPU 标杆端，以及 GPU 数据不能离开内网时的
部署方式。GPU 训练容器不安装 `torch_npu`、NPU driver 或完整 CANN 运行栈。

## 1. 推荐部署

```text
NPU 服务器
  Ascend PyTorch Profiler DB + msProbe target dump
                         |
                         | 单向同步进入 GPU 内网
                         v
GPU 内网共享目录 <--- Nsys SQLite + msProbe golden dump
        |
        +--- CUDA 训练容器：TorchTitan、Nsys、基础 msProbe
        +--- 离线分析环境：msprof-analyze、基础 msProbe compare
```

性能和精度比较都在 GPU 内网完成。GPU 原始 profile/tensor 不需要传出；只要安全
策略允许 NPU 交付件进入 GPU 内网，就能形成完整比较。如果连输入文件也不允许
进入，自动跨设备比较在信息上不可实现，只能分别阅读两端摘要。

## 2. GPU 训练与 Nsys 采集环境

先使用服务器已有且已经验证的 CUDA、PyTorch 和 NCCL，不要为了安装 MindStudio
工具升级这些核心包：

```bash
conda activate torchtitan

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda runtime:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY

nsys --version
nsys status --environment
```

源码安装三仓时，GPU 环境不安装 `torch_npu`，Turbo 的 NPU patch 也不是 GPU
采集依赖：

```bash
cd /workspace/yyb/torchtitan
python -m pip install -e . --no-deps

cd /workspace/yyb/torchtitan-test
python -m pip install -r requirements-test.txt
python -m pip check
```

运行标准 GPU 性能采集：

```bash
cd /workspace/yyb/torchtitan-test
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_nsys/performance_benchmark.py \
  --probe --topology single

python tests/glm5_2_nsys/performance_benchmark.py \
  --probe --topology ddp2
```

默认会采集 `cuda,nvtx,osrt,cublas,cudnn`，并启用 PyTorch function/shape 和
autograd NVTX。`--probe` 随后通过 `nsys export` 生成 `.sqlite`；跨平台性能校准
读取 SQLite，而不是 `.nsys-rep`。具体参数见
[`tests/glm5_2_nsys/README.md`](../../glm5_2_nsys/README.md)。

## 3. GPU 侧基础 msProbe

GPU 侧 Module/API statistics 或 tensor dump 不依赖 CANN。不要在纯 GPU 环境构建
`nan_check`、`xor_checksum`、`aclgraph_dump` 等 NPU 专属模块。当前仓库默认的
MindStudio bootstrap 面向完整 NPU 标准环境，不能原样用于纯 GPU 采集端。

源码安装基础包：

```bash
export MS_TOOLS=/workspace/yyb/mindstudio-tools
mkdir -p "${MS_TOOLS}"
cd "${MS_TOOLS}"

git clone https://gitcode.com/Ascend/msprobe.git
cd msprobe
python -m pip install uv
python build.py
python -m pip install --force-reinstall artifacts/mindstudio_probe*.whl

python - <<'PY'
import msprobe
print("msprobe:", msprobe.__file__)
PY
msprobe --help >/dev/null
```

探索阶段可跟踪官方分支；正式实验必须 checkout 已验证的 tag/完整 commit，并记录
`git rev-parse HEAD`。如果公司内网禁止 clone，在可进入该内网的受控环境构建基础
wheel，然后连同 commit、SHA256 和依赖清单一起传入。

检查 GLM GPU 精度采集环境：

```bash
cd /workspace/yyb/torchtitan-test
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope accuracy-gpu
```

GPU 支持通用 Module/API dump；NPU Kernel 级 dump、NPU 溢出寄存器检查以及 ACL
Graph dump 不属于 GPU 端能力。

## 4. GPU 内网的离线分析环境

最简单的部署是在 GPU 服务器创建第二个 Python 环境。它只读取文件，不参与 CUDA
训练：

```bash
conda create -n glm5-ms-analysis python=3.12 pip -y
conda activate glm5-ms-analysis

export MS_TOOLS=/workspace/yyb/mindstudio-tools
cd "${MS_TOOLS}"
git clone https://gitcode.com/Ascend/msprof-analyze.git
cd msprof-analyze
python -m pip install -r requirements/build.txt
python -m pip install --no-deps -e .

msprof-analyze --help
msprof-analyze cluster -m calibrate_npu_gpu --help
```

`calibrate_npu_gpu` 是离线 recipe，不需要访问 GPU/NPU 设备。若所选发行包在启动
时仍强制加载 CANN 共享库，不要把 CANN 注入 CUDA 训练环境；应在同一 GPU 服务器
上建立隔离的 CANN 工具容器/环境，挂载同一个只读数据目录，并在那里运行分析。

基础 msProbe wheel 也安装到该分析环境，用于精度 compare：

```bash
conda activate glm5-ms-analysis
python -m pip install \
  /workspace/yyb/mindstudio-tools/msprobe/artifacts/mindstudio_probe*.whl
msprobe compare --help
```

## 5. 性能数据汇合与比较

需要汇合：

```text
NPU: ascend_pytorch_profiler_*.db（必须包含 MSTX module 层级）
GPU: profile.sqlite（必须包含 NVTX module 层级和 CUDA kernel）
```

执行：

```bash
conda activate glm5-ms-analysis

msprof-analyze cluster -m calibrate_npu_gpu \
  --profiling_path /secure/inbound/npu_profile \
  --baseline_profiling_path /workspace/yyb/torchtitan-test/nsys_runs/PROFILE.sqlite \
  --output_path /workspace/yyb/analysis/calibrate_npu_gpu \
  --export_type text \
  --dump_intermediate_results
```

输出重点是 `compare_profile_report_<rank>.xlsx`。没有 NVTX/MSTX 时仍可能读取
Kernel，但不能可靠建立 GPU/NPU Module 对应关系；这不是补一个文件名就能修复的。

## 6. 精度数据汇合与比较

GPU 和 NPU 必须分别用同一 fixture、seed checkpoint、token plan、step/rank 范围和
msProbe 层级采集。把 NPU target dump 单向同步进入 GPU 内网后执行：

```bash
conda activate glm5-ms-analysis

# 单 rank
msprobe compare \
  -tp /secure/inbound/npu_dump/step0/rank0/dump.json \
  -gp /workspace/yyb/gpu_dump/step0/rank0/dump.json \
  -o /workspace/yyb/analysis/msprobe_compare \
  -da

# 多 rank：参数指向 rank 目录的共同父级 step0
msprobe compare \
  -tp /secure/inbound/npu_dump/step0 \
  -gp /workspace/yyb/gpu_dump/step0 \
  -o /workspace/yyb/analysis/msprobe_compare_multi \
  -da
```

采集完成并不证明输入一致；fixture/checkpoint/token plan 的 digest 必须先通过本项目
manifest 校验，之后才能把差异解释为设备数值差异。

## 7. 内网交付最小集合

从 NPU 侧传入 GPU 内网：

```text
performance/
  ascend_pytorch_profiler_*.db
  capture manifest、版本与源码 commit
accuracy/
  step*/rank*/construct.json
  step*/rank*/dump.json
  step*/rank*/stack.json
  tensor 文件（仅 task=tensor 时）
  fixture/config/checkpoint/token-plan digest
```

不要只传 HTML。跨平台分析需要原始 DB/dump 中的 Module、API、Kernel 和数值统计。
正式传输包应附 SHA256；到达 GPU 内网后先校验，再运行 compare。

## 8. 官方依据

- [NPU/GPU 性能拆解比对](https://github.com/Ascend/msprof-analyze/blob/master/docs/zh/advanced_features/calibrate_npu_gpu_instruct.md)
- [msProbe PyTorch 快速入门](https://github.com/Ascend/msprobe/blob/master/docs/zh/quick_start/pytorch_quick_start.md)
- [msProbe PyTorch 精度比对](https://github.com/Ascend/msprobe/blob/master/docs/zh/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [msProbe 安装指南](https://github.com/Ascend/msprobe/blob/master/docs/zh/msprobe_install_guide.md)
- [Nsight Systems 文档](https://docs.nvidia.com/nsight-systems/)

