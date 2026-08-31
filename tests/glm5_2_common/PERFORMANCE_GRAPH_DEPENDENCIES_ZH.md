# 性能与图模式环境、外部工具和依赖总表

本文是 `glm5_2_performance`、`glm5_2_graph` 和
`glm5_2_combination` 的统一依赖清单。安装、升级或迁移环境时，以本文为入口，
不要分别从三个实验 README 中猜依赖。

这里把工具分成五类：训练与编译运行时、性能数据采集、离线诊断、交互可视化、
图编译诊断。一个工具出现在本文不代表每次实验都必须安装；“必需范围”一列说明
它在哪个阶段真正参与执行。

## 1. 一张图理解工具链

```text
TorchTitan training
  |
  +-- eager -----------------------------------------------------+
  |                                                              |
  +-- torch.compile                                              |
       Dynamo -> FX -> AOTAutograd -> Inductor -> Triton-Ascend  |
            |       TORCH_TRACE / TORCH_COMPILE_DEBUG            |
            |                     |                               |
            |                  tlparse + FX/IR/code               |
            |                                                     |
            +---------------- runtime -----------------------------+
                                  |
                         torch_npu.profiler
                                  |
          +-----------------------+------------------------+
          |                       |                        |
     parsed text/DB          trace_view.json          memory/stack
          |                       |                        |
    msprof-analyze       MindStudio / Perfetto      flame graph / HTML
```

- `torch_npu.profiler` 和 `msprof` 是采集/解析链路，不是同一份报告的两个名称。
- `msprof-analyze` 消费已有数据做 advisor、compare、cluster 等诊断，不负责训练。
- MindStudio Insight 是官方交互式可视化器，不负责启动 TorchTitan。
- `tlparse` 看“编译器为什么分图、重编译、失败”；MindStudio 看“编译后的任务实际
  如何在 Host/NPU 上运行”。二者解决的问题不同。
- TensorBoard 看长时间训练标量，不是 NPU kernel 时间线，也不是编译图浏览器。

## 2. 必需运行时

| 依赖 | 归属与来源 | 本项目用途 | 必需范围 | 安装/版本原则 |
|---|---|---|---|---|
| Python | [Python](https://www.python.org/) | 三仓源码、实验编排和分析脚本 | 所有实验 | 使用 Torch/TorchNPU 支持的 Python 版本 |
| PyTorch | [pytorch/pytorch](https://github.com/pytorch/pytorch) | Trainer、DTensor、Dynamo、AOTAutograd、Inductor、Profiler 公共 API | 所有实验 | 必须与 TorchNPU 配套，不单独随意升级 |
| Ascend Extension for PyTorch | [Ascend/pytorch](https://gitcode.com/Ascend/pytorch) | `torch_npu`、NPU backend、Ascend PyTorch Profiler、NPUGraph、NPU Inductor | NPU 性能和图模式 | 严格使用官方兼容矩阵对应版本 |
| CANN Toolkit | [昇腾软件安装与兼容文档](https://www.hiascend.com/document) | runtime、算子库、HCCL、profiling 底层能力、`msprof` | 所有 NPU 实验 | 与驱动、固件、TorchNPU 匹配；不要只看 Python 包版本 |
| Triton | [triton-lang/triton](https://github.com/triton-lang/triton) | Inductor/Triton 公共编译接口 | Inductor 图模式 | 由配套环境确定 |
| Triton-Ascend | [Ascend/triton-ascend](https://gitcode.com/Ascend/triton-ascend) | 将 Triton kernel 编译到 Ascend NPU | NPU `backend=inductor` | 必须与 TorchNPU、CANN 和编译器工具链匹配 |
| TorchTitan | [pytorch/torchtitan](https://github.com/pytorch/torchtitan) | 设备无关训练、模型和并行框架 | 所有实验 | 源码安装；记录 commit |
| TorchTitanTurbo | [Ascend/TorchTitanTurbo](https://gitcode.com/Ascend/TorchTitanTurbo) | NPU patch、图模式兼容和 `torch_npu.profiler` 接入 | NPU 实验 | 源码安装；与 TorchTitan commit/API 对照 |

这组依赖不是本次“额外可视化包”，但它决定所有产物是否可信。至少执行一次：

```bash
python - <<'PY'
import torch
import torch_npu
import torchtitan
import torchtitanturbo

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("npu available:", torch.npu.is_available())
print("compile backends:", torch.compiler.list_backends())
print("torchtitan:", torchtitan.__file__)
print("torchtitanturbo:", torchtitanturbo.__file__)
PY

python -m pip show triton triton-ascend
```

如果 `inductor`/`npugraphs` 未出现在 backend 列表中，先修环境，不能通过实验代码
静默降级来声称图模式已跑通。

### 2.1 已有 NPU 训练环境后，一次补齐全部分析功能

下面假设 Torch、TorchNPU、CANN、Triton-Ascend、TorchTitan 和 TorchTitanTurbo 已按
兼容矩阵安装，当前目录是 `torchtitan-test`。`GLM5_TOOL_ROOT` 必须放在有足够空间的
数据盘，不要放在 TorchTitan 源码目录或容量很小的容器 overlay 中。

```bash
conda activate torchtitan

# Change this path to the persistent large-volume directory on the server.
export GLM5_TOOL_ROOT=/workspace/xxx/glm5-tools
mkdir -p "$GLM5_TOOL_ROOT"

# System readers used by source checkout, archive extraction and the portable
# FlameGraph fallback. Choose the command matching the host OS.
# sudo dnf install -y git perl unzip
# sudo apt-get update && sudo apt-get install -y git perl unzip

# Python report readers: matplotlib, TensorBoard and tlparse.
python -m pip install -r \
  tests/glm5_2_performance/requirements-visualization.txt

# Official Ascend offline analyzer. For a frozen delivery environment, replace
# -U with the version matched to the installed CANN release.
python -m pip install -U msprof-analyze

# Official MindStudio Insight source contains the standalone DB -> HTML flame
# graph generator. The GUI itself is installed separately below.
test -d "$GLM5_TOOL_ROOT/msinsight/.git" || \
  git clone https://gitcode.com/Ascend/msinsight.git \
    "$GLM5_TOOL_ROOT/msinsight"

# Portable folded-stack SVG renderer.
test -d "$GLM5_TOOL_ROOT/FlameGraph/.git" || \
  git clone https://github.com/brendangregg/FlameGraph.git \
    "$GLM5_TOOL_ROOT/FlameGraph"

export TORCHTITAN_MSINSIGHT_FLAMEGRAPH=\
"$GLM5_TOOL_ROOT/msinsight/scripts/flame_graph/flamegraph.py"
export TORCHTITAN_FLAMEGRAPH_PL=\
"$GLM5_TOOL_ROOT/FlameGraph/flamegraph.pl"

# msprof-analyze installed by pip is normally on PATH. This optional override
# is only needed when the executable is installed elsewhere.
# export TORCHTITAN_MSPROF_ANALYZE=/absolute/path/to/msprof-analyze
export TORCHTITAN_MSPROF_ANALYZE_WORKERS=1

python -m pip check
tensorboard --version
tlparse --help
msprof-analyze --help
perl --version
python "$TORCHTITAN_MSINSIGHT_FLAMEGRAPH" --help
```

这些 `export` 只对当前 shell 生效。长期使用可以写入该 conda 环境的 activation script，
但不要覆盖系统级 CANN 环境脚本。

### 2.2 MindStudio Insight GUI

GUI 不通过 `pip` 安装。推荐在用于阅读报告的 Windows 本机安装：

1. 本项目服务器使用 CANN 9.1.0，因此选择与之配套的 **MindStudio Insight 26.1.0**，
   不使用 26.0.0；从 [MindStudio 26.1.0 官方文档与下载入口](https://www.hiascend.com/document/detail/zh/mindstudio/2610/index/index.html)
   下载 `MindStudio-Insight_26.1.0_win.exe` 和校验文件；
2. 校验下载文件后运行安装程序；
3. 把服务器上的完整 `*_ascend_pt`/cluster 目录同步到本机；
4. 在 MindStudio Insight 中选择 profile 根目录导入。

Linux GUI 软件包下载完成后的基本命令是：

```bash
unzip MindStudio-Insight_26.1.0_linux-x86_64.zip \
  -d "$GLM5_TOOL_ROOT/MindStudio-Insight"
"$GLM5_TOOL_ROOT/MindStudio-Insight/MindStudio-Insight"
```

AArch64 使用对应 `linux-aarch64.zip`。无桌面服务器通常不值得为 GUI 安装 VNC/X11；
直接在服务器采集、在 Windows 本机可视化更简单。

现有仓库历史中，已经生成并同步过 MindStudio Insight 可导入的 `*_ascend_pt`、
`trace_view.json` 和 Ascend DB，也使用过 `msprof-analyze` 及项目 HTML 做离线分析；
但没有保存 MindStudio Insight GUI 的导入记录、截图或导出 workspace。因此准确的说法是：
**此前准备好了 Insight 输入，但尚未把 GUI 作为正式实验步骤使用。** 后续首次使用时，
应在实验 `readme.md` 中记录 Insight 版本、导入的 profile 根目录和关键截图/结论；GUI
本身不参与训练 PASS/FAIL 判定。

#### 2.2.1 Windows 本机 Conda 阅读环境

MindStudio Insight 的 Windows `.exe` 与下面的 Conda 环境相互独立：前者负责官方
Ascend 交互界面，后者用于在本机启动 TensorBoard、重新运行 `tlparse`/matplotlib
分析脚本和下载 Release。已经由服务器生成的项目 HTML、tlparse HTML、火焰图 HTML/SVG
只需要浏览器，不要求 Windows 再安装对应生成工具。Windows 本机不运行 NPU 训练，
因此**不要**在这个环境安装 CANN、TorchNPU、
Triton-Ascend 或 `msprof-analyze`；采集、CANN 解析和 advisor/cluster/compare 都留在
Linux NPU 服务器执行。

下面命令在 Windows PowerShell 中执行。Python 3.12 与当前三仓实验环境一致；这个环境
只读取/重新解析已有产物，不要求安装 TorchTitan。如果只看同步好的静态 HTML，可以
跳过整个 Conda 环境，只安装浏览器和 MindStudio Insight：

```powershell
conda create --name glm5-analysis python=3.12 pip -y
conda activate glm5-analysis

Set-Location D:\yyb\torchtitan-test
python -m pip install --upgrade pip
python -m pip install -r .\tests\glm5_2_performance\requirements-visualization.txt
python -m pip check

python -c "import matplotlib, tensorboard; print('matplotlib', matplotlib.__version__); print('tensorboard', tensorboard.__version__)"
tensorboard --version
tlparse --help
```

如果还没有 GitHub CLI，可单独安装并登录；它不属于 Conda/Python 依赖：

```powershell
winget install --id GitHub.cli -e
gh auth login
gh --version
```

下载服务器发布的分析包并在本机阅读：

```powershell
conda activate glm5-analysis
Set-Location D:\yyb\torchtitan-test

$Experiment = "migration-cuda-npu-single-bf16-random-s5000-b64-seq128-seed61"
python .\release_artifacts.py download $Experiment

# Keep this terminal open, then browse http://127.0.0.1:8000/.
python -m http.server 8000 --directory D:\yyb\torchtitan-test
```

若目标目录已有同名文件并确认要用 Release 覆盖，给 download 增加 `--overwrite`。
读取长期训练曲线和手工重建编译诊断的示例。二者也可以在服务器执行，再通过 SSH
端口转发访问 TensorBoard、或把 tlparse 输出目录同步到 Windows；不要在两端重复运行：

```powershell
tensorboard --logdir D:\yyb\torchtitan-test\performance_runs --port 6006

tlparse `
  D:\yyb\torchtitan-test\graph_runs\EXPERIMENT\TOPOLOGY\graph_visualization\rank0\dedicated_log_torch_trace.log `
  -o D:\yyb\torchtitan-test\graph_reports\tlparse-manual-rank0
```

`tlparse` 的输入文件名以实际产物为准；通常框架 compare/analyze 已自动生成
`graph_visualization/tlparse/`，本机不必重复解析。Perfetto 直接用浏览器打开
`trace_view.json`。MindStudio Insight 则启动独立 Windows 应用并导入从 `full` Release
恢复的完整 profile 根目录，而不是使用上述 Python HTTP 服务。

### 2.3 Perfetto

在线方式无需安装：浏览器打开 [Perfetto UI](https://ui.perfetto.dev/) 后导入
`trace_view.json`。内部数据不能上传到不可信站点；隔离环境按
[google/perfetto](https://github.com/google/perfetto) 官方方式部署本地 UI。

### 2.4 全功能最小验证

先验证一张 NPU，不要一开始启动“所有 topology x 所有 preset”：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

# Every profiler collection policy is a separate capture and output directory.
python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology single \
  --preset all --analysis-tools all
```

确认 feature suite 后，才在八卡服务器执行完整拓扑矩阵：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
unset CUDA_VISIBLE_DEVICES

python tests/glm5_2_performance/profiler_benchmark.py \
  --probe --device npu --topology all \
  --preset all --analysis-tools all
```

后一个命令是大型实验矩阵：每个 `topology x preset` 都是独立训练，不是一次 capture。
它会产生大量 DB/JSON/stack，请先检查数据盘空间。

图编译、正式精度和性能同时开启时，使用 combination，不把编译参数注入独立
performance runner：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
unset CUDA_VISIBLE_DEVICES

COMMON_ARGS="--topology single --objectives precision,performance \
--reference-graph eager --candidate-graph inductor \
--compiler-diagnostics --profiler-preset runtime"

tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --data --data-device npu $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture reference $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate $COMMON_ARGS
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --compare --require-all $COMMON_ARGS
```

combination 一次只选择一个 profiler preset；需要遍历所有采集策略时运行上面的独立
performance `--preset all`，避免一次正式精度 capture 被多套高开销采集重复扰动。

## 3. Python 侧新增依赖

这些包集中写在：

```text
tests/glm5_2_performance/requirements-visualization.txt
```

一键安装：

```bash
python -m pip install -r \
  tests/glm5_2_performance/requirements-visualization.txt
```

| 包 | 官方来源 | 为什么使用 | 是否影响训练数学 |
|---|---|---|---|
| `matplotlib` | [matplotlib/matplotlib](https://github.com/matplotlib/matplotlib) | 在无 GUI 的 compare/analyze 节点生成静态统计图 | 否，只读产物 |
| `tensorboard` | [tensorflow/tensorboard](https://github.com/tensorflow/tensorboard) | 打开 TorchTitan event 文件，查看 loss、grad norm、吞吐、MFU、显存等随 step 的曲线 | 否，只读 event |
| `tlparse` | [meta-pytorch/tlparse](https://github.com/meta-pytorch/tlparse) | 解析 PyTorch 官方 `TORCH_TRACE`，生成交互式编译报告 | 否，只读结构化编译 trace |

验证：

```bash
python -m pip show matplotlib tensorboard tlparse
tensorboard --version
tlparse --help
```

`tlparse` 不是随便选择的小众画图包。PyTorch 官方的
[`torch.compile` troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
明确建议：大模型编译问题先看 `TORCH_TRACE + tlparse`，已知具体子系统后再用
`TORCH_LOGS` 做细粒度调试。

## 4. Ascend 性能工具

### 4.1 Ascend PyTorch Profiler：默认采集入口

- 来源：[Ascend PyTorch Profiler 用户指南](https://gitcode.com/Ascend/pytorch/blob/master/docs/zh/developer_notes/ascend_pytorch_profiler_user_guide.md)
- 实现：[torch_npu/profiler](https://github.com/Ascend/pytorch/tree/master/torch_npu/profiler)
- 本项目接入：TorchTitanTurbo 把 TorchTitan profiler 生命周期映射到
  `torch_npu.profiler.profile`；test 仓库只编排 preset、参数、产物和报告。

它同时采集：

- Host CPU 上的 Python/PyTorch API；
- CANN 软件栈中的 AscendCL、GE、Runtime 等事件；
- NPU stream/task/kernel；
- HCCL 通信；
- 可选 shape、stack、module、memory、AI Core counter、L2、系统数据。

它依赖正确安装的 `torch_npu` 和 CANN，不需要再 `pip install` 一个同名包。

### 4.2 `msprof`：底层采集与离线解析能力

- 来源：[Ascend/msprof](https://gitcode.com/Ascend/msprof)
- 安装：通常随 CANN Toolkit，而不是通过本项目 requirements 安装。
- 本项目定位：保留为 profiler 底层环境和黑盒排障工具；日常 TorchTitan 采集
  首选 `torch_npu.profiler`，避免外层命令和训练内 schedule 重复控制。

检查：

```bash
which msprof
msprof --help
```

未解析的 `PROF_*` 原始目录要先经官方解析，才能得到 MindStudio 可读的 text/DB。
本框架的 `--offline-parse`/`--analysis-tools all` 负责触发对应流程。

#### 4.2.1 `msprof` 环境变量从哪里来

这里必须区分“CANN 安装环境”和“本项目实验开关”。CANN Toolkit 的
`set_env.sh`（或已经执行过该脚本的官方容器入口）负责提供 `msprof` 所需的基础环境：

- `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH` 等安装根路径；
- 把 `tools/profiler/bin` 和 CANN `bin` 加入 `PATH`，因此 shell 才能找到 `msprof`；
- 把 CANN、driver、opskernel 等目录加入 `LD_LIBRARY_PATH`；
- 把 CANN Python 和 TBE 路径加入 `PYTHONPATH`。

所以我们过去能直接执行 `msprof`，通常是因为容器镜像/登录脚本已经 source 了 CANN
环境，或者图模式 wrapper 显式 source 了对应版本的 `set_env.sh`。这不表示所有 profiler
选项都是 Toolkit 默认开启的。当前 CANN 9.1.0 的实际导出值和验证命令记录在
`tests/glm5_2_graph_debug/CANN_9_1_INSTALLATION.md`；日常应 source 脚本，不要手抄其中
的绝对路径。

| 名称 | 来源 | 含义 |
|---|---|---|
| `ASCEND_HOME_PATH`、`ASCEND_TOOLKIT_HOME`、`ASCEND_OPP_PATH`、`PATH`、`LD_LIBRARY_PATH`、`PYTHONPATH` | CANN `set_env.sh`/容器基础环境 | 让 runtime、算子库和 `msprof` 可发现 |
| `TORCHTITAN_MSPROF_ANALYZE` | torchtitan-test 可选覆盖 | 指向单独安装的 `msprof-analyze` 命令；不是 CANN 默认变量 |
| `TORCHTITAN_MSPROF_ANALYZE_WORKERS` | torchtitan-test | 限制离线分析并发；不改变采集或训练 |
| `GLM5_PERFORMANCE_*` | torchtitan-test | 在实验配置、runner 和 Turbo profiler 间传递采集策略 |
| `MSPROF_TX`、`MSTX` 及 domain include/exclude | TorchTitanTurbo/Ascend PyTorch Profiler 实验选项 | 选择用户 range 采集；不会因 source Toolkit 自动开启 |

`msprof-analyze` 是独立离线分析工具，`pip install msprof-analyze` 或源码安装后才会
出现；CANN `set_env.sh` 只解决 `msprof` 及其底层库，不能替代它。最小审计命令：

```bash
# This project's pinned CANN 9.1 environment. A standard Toolkit installation
# may instead expose /usr/local/Ascend/ascend-toolkit/set_env.sh.
source /usr/local/Ascend/cann-9.1.0/set_env.sh
command -v msprof
msprof --help
command -v msprof-analyze || true
env | grep -E '^(ASCEND_|TORCHTITAN_MSPROF|GLM5_PERFORMANCE|MSPROF_TX|MSTX)'
```

### 4.3 `msprof-analyze`：离线诊断

- 来源：[Ascend/msprof-analyze](https://gitcode.com/Ascend/msprof-analyze)
- 作用：对已有 profiler 数据执行 advisor、compare、cluster、通信瓶颈、集群时间等
  recipe；它不采集训练数据。
- 本项目使用：`--advisor`、`--cluster`、`--compare-baseline` 或
  `--analysis-tools all`。

安装方式应遵循仓库/对应 CANN 版本文档。安装后：

```bash
msprof-analyze --help
```

如果可执行文件不在 `PATH`：

```bash
export TORCHTITAN_MSPROF_ANALYZE=/absolute/path/to/msprof-analyze
export TORCHTITAN_MSPROF_ANALYZE_WORKERS=1
```

worker 只影响离线分析并发，不改变训练；多 rank 大 DB 先从 `1` 开始，避免分析
进程把内存/磁盘打满。

统一图模式 launcher 使用 `env -i` 建立可复现的最小环境，因此这两个覆盖变量必须由
`graph_env_common.sh` 显式透传。不能只在交互 shell 中用 `env`/`command -v` 判断；应检查
实际 launcher 子进程：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor command -- \
  /usr/bin/env | grep '^TORCHTITAN_MSPROF_ANALYZE'
```

如果外层变量存在而上述命令没有输出，问题属于 launcher 环境契约，不应误判为
`msprof-analyze` 未安装。

### 4.4 MindStudio Insight：官方交互可视化

- 来源：[Ascend/msinsight](https://gitcode.com/Ascend/msinsight)
- 官方教程：[System Tuning](https://gitcode.com/Ascend/msinsight/blob/master/docs/zh/user_guide/system_tuning.md)
- 作用：导入完整 `*_ascend_pt` 或集群数据库，交互查看 Timeline、Memory、
  Operator、Summary、Communication，并关联 Host API 与 NPU task。
- 安装：按 MindStudio Insight 发布包说明安装在有桌面的本机/分析机；训练容器不必
  安装 GUI。

MindStudio 不是 Python 包。推荐“服务器采集 -> 同步完整目录 -> 本机导入”。不要只
复制一张 `trace_view.json` 就期望看到完整通信/内存页。

### 4.5 MindStudio 官方 HTML 火焰图

- 来源：[msinsight/scripts/flame_graph](https://gitcode.com/Ascend/msinsight/tree/master/scripts/flame_graph)
- 输入：含 `PYTORCH_API` 和 `STRING_IDS` 表的
  `ascend_pytorch_profiler_<rank>.db`；采集时必须包含 CPU activity。
- 输出：自包含 `flamegraph.html`，支持线程切换、函数搜索、悬浮详情和点击缩放；
  分类 Python、PyTorch/framework、CANN。
- 限制：官方脚本当前限制输入 DB 不超过 10GB；HTML 会随 API 数量增大。

告诉框架脚本位置：

```bash
export TORCHTITAN_MSINSIGHT_FLAMEGRAPH=\
/path/to/msinsight/scripts/flame_graph/flamegraph.py

python tests/glm5_2_performance/profiler_benchmark.py \
  --analyze --device npu --topology single --preset flamegraph
```

也可以把环境变量指向包含 `flamegraph.py` 的目录。分析阶段会为每个 rank DB 写入
独立目录，并把 HTML 和生成日志加入性能报告。它只分析 Host CPU 调用层级；NPU
kernel 层级仍看 Timeline 和 folded NPU stack。

## 5. 通用/便携可视化工具

### 5.1 Perfetto

- 来源：[Perfetto UI](https://ui.perfetto.dev/) 和
  [google/perfetto](https://github.com/google/perfetto)
- 作用：浏览 `trace_view.json` 的时间线、slice、线程和异步事件。
- 安装：通常不需要，浏览器打开网页后导入文件；涉密/隔离环境可使用官方仓库自行
  部署，不应把内部 trace 上传到不可信站点。
- 边界：轻量、通用，但不理解 Ascend Summary/Communication/Memory 的专有语义。
  NPU 正式分析仍以 MindStudio Insight 为主。

### 5.2 Brendan Gregg FlameGraph：便携兜底

- 来源：[brendangregg/FlameGraph](https://github.com/brendangregg/FlameGraph)
- 作用：把 `export_stacks` 产生的 CPU/NPU folded stack 转成可缩放 SVG。
- 依赖：Git checkout + Perl；不是 Python 包。

```bash
git clone https://github.com/brendangregg/FlameGraph.git /opt/FlameGraph
export TORCHTITAN_FLAMEGRAPH_PL=/opt/FlameGraph/flamegraph.pl
perl --version
```

它与 MindStudio 官方 HTML 的区别：

- MindStudio HTML 从 Ascend DB 重建 Host API 层级，按线程查看；
- `flamegraph.pl` 消费框架导出的 folded CPU/NPU stack，便携且能保留 NPU 聚合栈；
- 两者都不是按真实先后排列的 Timeline。

### 5.3 TensorBoard

启动命令由报告自动给出，通常是：

```bash
tensorboard \
  --logdir /absolute/path/to/trainer_output/tensorboard \
  --port 6006 --bind_all
```

远程服务器建议用 SSH 端口转发，而不是把 6006 暴露到公网：

```bash
ssh -L 6006:127.0.0.1:6006 user@server
```

TensorBoard 负责跨 step 的 loss/grad norm/吞吐/MFU/内存曲线。它不替代
MindStudio 的微秒级时间线，也不显示 Dynamo graph break。

## 6. 图编译诊断工具

### 6.1 `TORCH_TRACE + tlparse`

- 来源：[PyTorch troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)、
  [meta-pytorch/tlparse](https://github.com/meta-pytorch/tlparse)
- 作用：以编译 frame 为单位展示 graph break、recompile、guards、错误、FX 图和
  编译产物索引。
- 本项目：`--compiler-diagnostics` 给每个 rank 设置独立 `TORCH_TRACE`；compare
  自动执行 `tlparse`。
- 隐私：structured trace 不含模型权重，但会包含源码、文件路径、错误和调用信息，
  外发前必须审阅。

### 6.2 `TORCH_COMPILE_DEBUG`

- 来源：[PyTorch `torch.compile` 调试文档](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)、
  [Ascend 图编译 Debug 信息保存](https://www.hiascend.com/document/detail/zh/Pytorch/2600/modthirdparty/torchairuseguide/docs/zh/ascend_ir/features/basic/debug_save.md)
- 作用：让后端 dump 可读 FX、融合前/后 IR、生成代码、编译日志。
- 本项目：`--compiler-diagnostics` 设置每 rank 独立的
  `TORCH_COMPILE_DEBUG_DIR`，避免分布式进程覆盖同一个目录。

`torch_compile_debug` 是 PyTorch/后端自动生成的诊断目录，不是本项目凭空构造的
报告，也不是 compiler cache。框架只索引与当前 run 绑定的受控目录。

### 6.3 `TORCH_LOGS`

- 来源：[PyTorch logging 文档](https://docs.pytorch.org/docs/stable/logging.html)
- 作用：在已经知道问题层级时查看 `dynamo`、`guards`、`recompiles`、`graph_breaks`、
  `inductor` 等细粒度文本日志。
- 使用原则：先看 tlparse 总览，再针对单个问题开启；不要长期给八卡全 rank 开最大
  verbosity。

### 6.4 Ascend 专有图产物

- NPU Inductor 可通过其配套版本的配置，例如
  `INDUCTOR_ASCEND_DUMP_FX_GRAPH`，dump Ascend lowering 看到的 FX；是否可用取决于
  当前 TorchNPU 版本。
- MindStudio Insight 可导入 ACLGraph 构图过程 JSON，在 Timeline 中展示 Record/Wait
  等图执行关系；它是 NPUGraph/ACLGraph 运行时图，不等价于 Dynamo 的 FX DAG。
- 本框架默认保存 PyTorch 官方诊断产物；版本专有环境变量只有在当前 TorchNPU 文档
  明确支持且实验身份记录后才启用，不能硬编码为所有版本的默认值。

## 7. 调研过但不是当前必需依赖的图工具

| 工具 | 来源 | 适合什么 | 为什么不是本框架主入口 |
|---|---|---|---|
| FX `FxGraphDrawer` | [PyTorch source](https://github.com/pytorch/pytorch/blob/main/torch/fx/passes/graph_drawer.py) | 用 Graphviz 画一张已持有的 FX `GraphModule` | 需要在 backend 回调中拿到具体 FX 图；不汇总多 frame、guards、recompile |
| Netron | [lutzroeder/netron](https://github.com/lutzroeder/netron) | ONNX/TorchScript 等序列化模型结构 | 不理解运行时 Dynamo/AOT/Inductor 多图 |
| torchview | [mert-kurttutan/torchview](https://github.com/mert-kurttutan/torchview) | `nn.Module` 前向模块/张量结构 | 不是 compiler IR，也不显示后端融合 |
| torchviz | [szagoruyko/pytorchviz](https://github.com/szagoruyko/pytorchviz) | autograd backward DAG | 不解释 graph break 或 kernel codegen |
| TensorBoard Graph | [TensorBoard Graphs](https://www.tensorflow.org/tensorboard/graphs) | 静态计算图教学/浏览 | PT2 编译链会产生多个图和多层 IR，覆盖不完整 |
| Nsight Systems | [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) | CUDA 时间线和 NVTX | GPU 路径预留；当前 NPU 实验不依赖 |

主流训推框架也采用“框架 profiler + 硬件时间线 + 有界 step/rank + 内存工具”的
组合，而不是一个万能图查看器。例如
[Megatron Bridge profiling](https://docs.nvidia.com/nemo/megatron-bridge/nightly/training/profiling.html)
分别接入 Nsight Systems 和 PyTorch Profiler，并要求选择采集 step/rank。NPU 上对应的
主链是 Ascend PyTorch Profiler + msprof-analyze + MindStudio Insight；编译器问题再加
tlparse/FX/IR/code。

## 8. 最小、完整和离线安装方案

### 8.1 只跑训练/图模式

使用已配套的 Torch、TorchNPU、CANN、Triton-Ascend 和三仓源码。不需要
TensorBoard、tlparse、matplotlib、MindStudio GUI。

### 8.2 服务器完整采集和自动报告

```bash
python -m pip install -r \
  tests/glm5_2_performance/requirements-visualization.txt

export TORCHTITAN_MSPROF_ANALYZE=/path/to/msprof-analyze
export TORCHTITAN_MSINSIGHT_FLAMEGRAPH=/path/to/flamegraph.py
export TORCHTITAN_FLAMEGRAPH_PL=/path/to/FlameGraph/flamegraph.pl
```

其中后两项是可选的：没有它们也能完成 capture，报告会明确显示哪种渲染器未就绪。

### 8.3 本地阅读机

建议安装 MindStudio Insight、TensorBoard 和浏览器。同步：

- 完整 `*_ascend_pt` 目录或统一 DB；
- `performance_reports`/`combination_reports`；
- `graph_visualization` 中的 tlparse、FX/IR/code；
- TensorBoard event 文件；
- 需要时同步 memory timeline 和 flame graph HTML/SVG。

Windows 的可复现 Conda 安装、GitHub CLI 下载、HTTP 阅读、TensorBoard 和 `tlparse`
命令见 [2.2.1 Windows 本机 Conda 阅读环境](#221-windows-本机-conda-阅读环境)。

只做报告阅读、Codex 分析和文档整理时，直接生成分析包：

```bash
python release_artifacts.py upload <experiment> --content analysis
```

该模式包含上述已解析/已渲染结果，不包含 checkpoint、原始 CANN 采集树、trainer state
和编译缓存。需要在 Windows MindStudio 中导入完整原始 profile，或在另一台机器重新运行
官方解析时，改用无损包：

```bash
python release_artifacts.py upload <experiment> --content full
```

Perfetto 可以直接打开分析包中的 trace JSON；MindStudio 完整重放应使用 `full` 包中的
profile 根目录。Git 只负责轻量 `*_reports`，这两种 Release 包负责报告目录之外的数据。

## 9. 环境验收清单

```bash
# NPU runtime and graph backend
python - <<'PY'
import torch
import torch_npu
print(torch.__version__, torch_npu.__version__)
print(torch.npu.is_available())
print(torch.compiler.list_backends())
PY

# Optional Python readers
python -m pip check
tensorboard --version
tlparse --help

# Ascend collection/analysis
which msprof
msprof --help
msprof-analyze --help

# Portable flame graph fallback
perl --version
test -f "$TORCHTITAN_FLAMEGRAPH_PL"

# Official MindStudio HTML flame graph
test -f "$TORCHTITAN_MSINSIGHT_FLAMEGRAPH"
python "$TORCHTITAN_MSINSIGHT_FLAMEGRAPH" --help
```

最后用 10-step probe 验证，而不是直接启动高开销八卡 `all`：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4

python tests/glm5_2_performance/profiler_probe.py \
  --probe --device npu --topology single --preset overview
```

## 10. 升级规则和安全边界

1. Torch、TorchNPU、CANN、Triton-Ascend 必须按兼容矩阵整体升级。
2. 升级任何 profiler API 后，审计 performance、graph、combination、Release 和报告
   发现逻辑；对应关系见仓库根目录 `DEPENDENCY_AUDIT.md`。
3. Profiler 与编译 trace 可能包含源码路径、Python 调用栈、环境变量和命令。对外分享
   前清理敏感信息。
4. `--preset all` 是多次独立采集，不是把所有高开销选项塞进一个进程。
5. profiler-active 时间不能作为真实性能基线；优化 A/B 以 profiler-off 多次重复为准，
   active capture 只做归因。
6. GUI、离线诊断器和渲染脚本都不得修改训练数学或 PASS/FAIL 标准。
