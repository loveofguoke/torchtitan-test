# GLM-5.2 MindStudio 官方标准实验

本目录是 GLM-5.2 的独立 MindStudio 官方标准工作流。实验的阶段、工具、输入、
产物和判读方法均以 MindStudio 26.1 官方文档为准：

```text
训练问题复现与配置 CheckList
  -> msProbe 配置检查、Monitor V2、模块/API/编译精度采集与比较
  -> msProf 或 Ascend PyTorch Profiler 性能采集
  -> msprof-analyze advisor / cluster / compare
  -> MindStudio Insight Summary / Timeline / Communication / Operator / Memory
  -> 根据首个异常证据缩小范围并重新采集
  -> 修改后使用相同契约执行 A/B 验收
```

当前代码只位于 `torchtitan-test`。它通过源码安装的 `torchtitan` 和
`TorchTitanTurbo` 运行模型，不修改 TorchTitan 数学实现，也不在测试仓库
复制 NPU patch。

性能标准流程见
[docs/PERFORMANCE_WORKFLOW_ZH.md](docs/PERFORMANCE_WORKFLOW_ZH.md)。该入口默认使用
msProf 建立通用 CANN/NPU 采集；需要 PyTorch module、shape、stack、memory 和
schedule 窗口时切到 Ascend PyTorch Profiler。msOpProf、msMemScope 和服务化
Profiler 只登记正确用途，未完成目标服务器验证前不会伪造整网训练命令。

## 1. 当前实现状态

| 能力 | 状态 | 入口 |
| --- | --- | --- |
| 工具源码 checkout、安装计划、版本/路径 doctor | 已实现 | `tools/bootstrap_mindstudio_toolchain.py`、`toolchain.py` |
| 固定 token plan 和 seed checkpoint | 已实现 | 每个 benchmark 的 `--data` |
| GPU/NPU 模块级、API 级采集 | 已实现 | `migration_benchmark.py` |
| 官方离线 `msprobe compare` | 已实现 | `migration_benchmark.py --compare` |
| NPU eager/compile 模块前向与反向比较 | 已实现 | `compile_accuracy_benchmark.py` |
| GPU/NPU 训练前配置检查与逐 rank compare | 已实现 | `configuration_check_benchmark.py` |
| API 精度预检与两端预检结果比对 | 已实现 | `--precheck`、`--precheck-compare` |
| 训练状态监控 | 已实现，独立于短时 dump capture | `training_monitor_benchmark.py` |
| 分级图可视化与 TensorBoard 索引 | 已实现 | migration 完成 L0/mix capture 后执行 `--graph-visualize` |
| NPU 性能采集、分析、可视化 | 已实现标准入口 | `performance_benchmark.py`、`docs/PERFORMANCE_WORKFLOW_ZH.md` |
| 推理部署、算子生产交付 | 尚未纳入当前训练工作流 | 能力边界见官方文档矩阵 |

“命令成功”只说明官方工具阶段完成，不等于精度通过。最终判断必须阅读
官方结果中的 `Result`、`Err_Message` 和各项误差指标，并结合端到端训练结果。

## 2. 文档入口

第一次使用按以下顺序阅读：

1. 本 README：安装、命令、参数、目录与边界；
2. [源码安装与环境检查](docs/SOURCE_INSTALL_ZH.md)：环境分层、源码工具安装、doctor；
3. [GPU/NPU 模块与 API 精度流程](docs/ACCURACY_WORKFLOW_ZH.md)：官方五阶段、预检、指标与结果判读；
4. [eager/compile 精度流程](docs/COMPILE_ACCURACY_WORKFLOW_ZH.md)：single-pass、FSDP2、多卡限制；
5. [官方文档与能力矩阵](docs/OFFICIAL_DOCUMENTATION_MATRIX_ZH.md)：官方章节、GLM 入口、产物和支持状态逐项对应；
6. [官方工具链全景](docs/OFFICIAL_TOOLCHAIN_ZH.md)：训练、图编译、性能、推理、算子工具的职责；
7. [官方性能工作流](docs/PERFORMANCE_WORKFLOW_ZH.md)：msProf、Ascend PyTorch Profiler、msprof-analyze 与 Insight；
8. [集群分析操作与判读](docs/MSPROF_ANALYZE_CLUSTER_ZH.md)：cluster 参数、交付件、字段、Insight 页面和定位动作；
9. [输出与报告](docs/OUTPUTS_AND_REPORTS_ZH.md)：raw、artifact、report、同步与保留策略；
10. [服务器验收矩阵](docs/SERVER_VALIDATION_MATRIX_ZH.md)：从单卡最小闭环到 all topology 的真实验收顺序；
11. [官方实践学习与复现路线](docs/OFFICIAL_PRACTICE_ROADMAP_ZH.md)：按现象选择精度、性能、图编译、算子和推理工具；
12. [实施计划与边界](docs/IMPLEMENTATION_PLAN_ZH.md)：标准工作流的实施状态与验收边界。

官方权威入口：

- [MindStudio 文档总入口](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [msProbe 源码](https://gitcode.com/Ascend/msprobe)
- [PyTorch 精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_accuracy_compare_instruct.md)
- [PyTorch 编译精度比对指导](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msProbe/docs/zh/user_guide/accuracy_compare/pytorch_compile_accuracy_compare_instruct.md)

## 3. 环境安装

### 3.1 前置条件

GPU reference 服务器需要可用的 CUDA PyTorch 环境。NPU candidate 服务器需要
互相兼容的 driver、firmware、CANN、PyTorch 和 torch_npu。工具脚本不会替用户
升级这些核心组件。

三个仓库使用源码安装：

```bash
cd /data/yyb_hys/torchtitan
python -m pip install -e . --no-deps

cd /data/yyb_hys/TorchTitanTurbo
python -m pip install -e . --no-deps

cd /data/yyb_hys/torchtitan-test
python -m pip install -r requirements-test.txt
python -m pip check
```

路径可替换成实际挂载目录。GPU 端不要求安装 torch_npu；NPU 端必须能
`import torchtitanturbo` 并由其加载 NPU 适配。

### 3.2 源码安装官方工具

默认脚本在仓库外 clone `msprobe` 和 `msprof-analyze`。msProbe wheel 会按官方
方式包含 `tb_graph_ascend`，因此构建机还需官方推荐的 Node.js 20.19.3 与 npm
10.8.2。先查看完整计划：

```bash
cd /data/yyb_hys/torchtitan-test

python tools/bootstrap_mindstudio_toolchain.py \
  --root /data/yyb_hys/mindstudio-tools \
  --dry-run
```

确认后执行：

```bash
python tools/bootstrap_mindstudio_toolchain.py \
  --root /data/yyb_hys/mindstudio-tools

export TORCHTITAN_MINDSTUDIO_TOOLCHAIN_ROOT=/data/yyb_hys/mindstudio-tools
```

需要同时取得 Insight 源码中的辅助脚本时：

```bash
python tools/bootstrap_mindstudio_toolchain.py \
  --root /data/yyb_hys/mindstudio-tools \
  --include-optional
```

也可以用 `--tool msinsight` 只选择该可选源码；`--tool` 可重复。默认 bootstrap
只处理 lock 中标记为 default 的 msProbe 与 msprof-analyze。

`msprof` 默认使用 CANN 安装中自带的可执行程序；MindStudio Insight GUI 使用
与 CANN 数据格式兼容的官方桌面安装包。源码 checkout 便于阅读和记录 commit，
不等于已经安装 GUI。

当前 lock 默认跟踪官方 `master`，适合探索。正式交付前应把
`toolchain.lock.json` 中的 `ref`/`commit` 固定为验证过的 tag 或完整 commit，
否则以后不能严格复现实验。

bootstrap 成功后还会在仓库外工具根写入
`/data/yyb_hys/mindstudio-tools/toolchain.resolved.json`。lock 表达“计划使用什么”，
resolved manifest 记录“本次实际 checkout/build/install 了什么”，包括实际 HEAD、
构建产物和执行命令。正式实验必须同时归档二者；只保存浮动 `master` 名称或只保存
`pip show` 都不足以复现。

### 3.3 doctor

```bash
# GPU capture 端
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope accuracy-gpu

# NPU capture 端
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope accuracy-npu

# 同时检查 profiler 分析栈
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope full

# 只检查官方性能链路
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance
```

doctor 是只读检查，不会自动安装或升级依赖。主要检查：

- `msprobe` CLI 与 Python import；
- torch、torch_npu、CANN 环境；
- `msprof` 与 `msprof-analyze`；
- 可选的 `npu-smi`、Insight 源码和 flamegraph 脚本；
- 外部源码目录、Git commit 和 dirty 状态。

需要把同一信息写入其他自动化系统时，可运行：

```bash
python -m tests.glm5_2_mindstudio.toolchain metadata \
  --scope accuracy-npu --compact
```

也可以通过任一 benchmark 查看其工作流 doctor；最终可用性以当前版本的
`--help` 和 doctor 输出为准：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --doctor --doctor-device cuda

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --doctor --doctor-device npu
```

## 4. GPU/NPU 模块与 API 精度迁移

下面命令对应最常用的单卡标准流程。fixture 可以在 GPU 或 NPU 任一端生成，
但只生成一次并同步到另一端。

### 4.1 生成共享数据

在 GPU 端生成：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device cuda --topology single
```

或在 NPU 端生成：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device npu --topology single
```

同步生成的 `mindstudio_fixtures/<experiment-id>/`。它包含固定 token plan、
seed checkpoint、fixture manifest 和 generation ID。

### 4.2 GPU reference capture

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology single
```

### 4.3 NPU candidate capture

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology single
```

### 4.4 API 精度预检

精度预检不是 `msprobe compare` 的别名。它要求先完成相同 identity 的 L1 或
`mix` API capture，然后在各自设备服务器上把每个 `dump.json` 交给官方
`acc_check`：工具会为每个被采 API 构造单元测试并与 CPU 高精度标杆比较。
最后再把 GPU、NPU 两端的 details CSV 放到同一环境，用
`api_precision_compare` 比较两端预检结论。

先把 4.1～4.3 的 data、reference capture、candidate capture 命令都加上
`--level L1 --dump-task statistics`，建立一套新的 API 级实验；不能直接拿默认
L0 artifact 做预检。完成两端 L1 capture 后，在 GPU 端执行：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck reference --topology single \
  --level L1 --dump-task statistics
```

NPU 端：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck candidate --topology single \
  --level L1 --dump-task statistics
```

同步两端完整 capture artifact，以及各端预检产生的以下目录：

```text
mindstudio_artifacts/<experiment-id>/single/precision_precheck/reference-r1/
mindstudio_artifacts/<experiment-id>/single/precision_precheck/candidate-r1/
```

reference/candidate 预检属于 endpoint artifact，不在 report 根。两端数据汇合后执行：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck-compare --topology single \
  --level L1 --dump-task statistics
```

成功后直接阅读：

```text
mindstudio_reports/<experiment-id>/single/precision_precheck/compare-r1/
├── precheck_report.html       # 逐 step/rank 可点击入口
├── precheck_index.json        # 机器可读索引
└── stepN/rankN/official/      # 官方 result/details 与 summary
```

该 HTML 明确表示“两个端点各自相对 CPU 高精度标杆，再比较两端误差”，不是直接
比较两个完整训练进程的输出 tensor。

数据量大时可改用官方 `multi_acc_check`。例如在两个 device 上合计切成 16 份：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck candidate --topology single \
  --level L1 --dump-task statistics \
  --precheck-device-ids 0,1 --precheck-splits 16
```

这只并行执行离线 API 单测，不会把单卡模型训练自动改成两卡训练。预检的 device
数量、线程划分、JIT 和支持 API 均受所锁定 msProbe/设备版本约束。当前自动
`--precheck-compare` 要求每个端点、step、rank 恰好一个 details CSV；多 device
产生的多份 details 会全部保留，但不能静默选一份比较。需要自动闭环时先使用单
device 预检；否则必须明确合并/选择规则并人工运行官方 compare。

若一次很大的 `acc_check` 中断，官方支持从已有 result CSV 断点续检。当前 wrapper
要求一次只选择一个 topology、step、rank，并会自动寻找同目录同时间戳的 details
CSV：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck candidate --topology single \
  --level L1 --dump-task statistics --dump-step 0 --dump-ranks 0 \
  --precheck-resume-csv /absolute/path/accuracy_checking_result_<timestamp>.csv
```

源 result/details 不会被原地修改；项目先复制到当前 member 的 `official/`，再把
复制后的 result 传给官方 checker，并把两个源文件的绝对路径和 SHA-256 写入本次
operation identity。只允许续接同一 capture/config 的结果，不能用 resume 掩盖
实验 identity 变化。

### 4.5 离线模块/API 比较

把两端 `mindstudio_artifacts/<experiment-id>/single/` 同步到同一仓库后执行：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single
```

默认配置是 `task=statistics`、`level=L0`、msProbe `step0`。定位到可疑模块后，
可以建立新的 API 或真实 tensor generation。下面以 API 统计量为例，data、
reference、candidate 和 compare 都必须带同一组 override：

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

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology single \
  --level L1 --dump-task statistics
```

改变 `task`、`level` 或 `dump-step` 会改变实验 identity。reference 与 candidate
必须以完全相同的配置重采，不能把旧 reference 与新 candidate 混搭。真实 tensor
同理，把上述四条命令中的参数统一换成 `--level L0 --dump-task tensor`；其体积
明显更大。

### 4.6 分布式拓扑与 all

查看真实拓扑注册表：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py --list-topologies
```

指定一个分布式拓扑，例如：

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

一次运行所有已注册且 world size 不超过 8 的拓扑：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device cuda --topology all

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology all

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology all

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --compare --topology all
```

如果要对所有拓扑继续执行 API 精度预检，必须让 data、两端 capture 和以下三条
命令都使用同一组 `--level L1 --dump-task statistics`：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck reference --topology all \
  --level L1 --dump-task statistics

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck candidate --topology all \
  --level L1 --dump-task statistics

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --precheck-compare --topology all \
  --level L1 --dump-task statistics
```

前两条分别在 GPU/NPU capture 服务器运行并设置对应可见设备；第三条在同步完整
artifact 与 `precision_precheck` 结果后运行。

`all` 是接口能力，不代表所有 topology 已在当前工具/CANN 版本完成服务器验收。
先跑 `single` 和一个代表性分布式拓扑。`tensor × all rank × all topology`
可能产生巨量数据，默认不要这样采。

完整方法、指标和排障顺序见
[ACCURACY_WORKFLOW_ZH.md](docs/ACCURACY_WORKFLOW_ZH.md)。

### 4.7 分级图可视化

该阶段复用已经完成的 migration capture，不重新训练。两端必须使用同一个
fixture generation，并以 `--level L0` 或 `--level mix` 生成非空
`construct.json`。完整单卡流程如下；若前面已经完成相同 identity 的 L0 capture，
前三个 data/capture 阶段会按完整性标记安全跳过：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --data --data-device cuda --topology single --level L0
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture reference --topology single --level L0

unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --capture candidate --topology single --level L0

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --graph-visualize --topology single --level L0
```

最后一条只运行官方 `graph_visualize`，不重新训练。

指定分布式拓扑或全部拓扑：

```bash
python tests/glm5_2_mindstudio/migration_benchmark.py \
  --graph-visualize --topology fsdp8 --level L0

python tests/glm5_2_mindstudio/migration_benchmark.py \
  --graph-visualize --topology all --level L0
```

可选 `--fuzzy-match`、`--graph-overflow-check`、`--graph-progress-log`；
`--tensor-log` 只允许同 identity 的 tensor dump。官方 `.vis.db` 保存在 ignored
artifact 中，tracked report 只保存 hash 索引、runtime log 路径和下面这种启动
命令，不会自动启动阻塞服务。实际目录是：

```text
mindstudio_runs/<experiment-id>/<topology>/graph-visualize-r1/runtime.log
mindstudio_artifacts/<experiment-id>/<topology>/graph-visualize-r1/official/*.vis.db
mindstudio_reports/<experiment-id>/<topology>/graph-visualize-r1/{README.md,index.json}
```

启动命令为：

```bash
tensorboard --logdir <mindstudio_artifacts/.../graph-visualize-r1/official> \
  --bind_all
```

通过 VSCode Remote 转发端口后在本地浏览器打开 TensorBoard 的 Ascend Graph
插件。`analysis` Release 会保留已处理的 `.vis.db`，但仍省略 raw tensor；`full`
Release 用于无损迁移全部输入与中间数据。上传前必须审查数据库中的模块名、统计量
和源码路径。

## 5. NPU eager/compile 精度

该流程不是分别运行 eager 与 graph 两次。官方 `PrecisionChecker` 的
single-pass 在同一个 NPU 训练 step 中，对被标记模块使用相同输入比较 eager
与 compiled 前向输出和输入梯度。

### 5.1 单卡

```bash
export ASCEND_RT_VISIBLE_DEVICES=4

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --data --data-device npu --topology single

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology single

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology single
```

### 5.2 一个分布式拓扑

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology fsdp8

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology fsdp8
```

### 5.3 all

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --data --data-device npu --topology all

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --capture candidate --topology all

python tests/glm5_2_mindstudio/compile_accuracy_benchmark.py \
  --compare --topology all
```

当前默认 backend 是 `inductor`，并按 `Glm5TransformerBlock` 标记模块。可用
backend 必须以目标服务器的 `torch.compiler.list_backends()` 和最小 smoke 为准，
不能仅因名字出现在列表中就认定训练可用。

重要限制：

- 训练/FSDP2 必须使用 single-pass，不能 `deepcopy` 模型；
- capture 会为每个 rank 写 `compile_coverage_rank<rank>.json`；每个
  rank/model-part 必须实际拥有至少一个 `Glm5TransformerBlock`，否则直接失败；
- 每个 part CSV 必须至少包含一条非 SKIP 的模块 `fwd_output`，以及一条
  `grad_input` 或 `grad_output`；只有 CSV 表头或合成的 `LOSS` PASS 行不能作为
  模块级 eager/compiled 对齐证据；
- 每个 rank 独立生成官方 CSV，项目层只做保留字段的汇总；
- single-pass 不执行完整 eager 全网，`loss_eager=NaN` 是预期行为；
- 该 checker 使用 `fullgraph=False` 做模块级诊断，不等价于 TorchTitan 正式
  `fullgraph=True` 图训练验收；
- 不要同时给该命令开启 TorchTitan 原生 `--compile.enable`，避免双重编译；
- 正式结论还要跑现有 graph/combination 的多 step 精度与编译诊断。

完整说明见
[COMPILE_ACCURACY_WORKFLOW_ZH.md](docs/COMPILE_ACCURACY_WORKFLOW_ZH.md)。

## 6. 配置检查

官方标准精度流程把“训练前配置检查”放在 dump/compare 之前。当前仓库使用
`ConfigChecker` dynamic 模式为每个 rank 生成配置包，再逐 rank 调用官方
`msprobe config_check` 比较。先生成与 migration 相同语义的共享 fixture：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --data --data-device cuda --topology single
```

然后执行两端 capture 和离线 compare：

```bash
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --capture reference --topology single

export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --capture candidate --topology single

python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --compare --topology single
```

配置检查只回答环境、参数、模型等配置是否存在影响精度的差异，不替代模块/API
数值比较。分布式和 `all` 使用相同的 `--topology fsdp8`/`--topology all`
形式；每个 rank 分别生成和比较配置包。API 精度预检已经由 migration 的
`--precheck`/`--precheck-compare` 独立编排。

完整 all-topology 配置检查命令如下：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --data --data-device cuda --topology all
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --capture reference --topology all

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --capture candidate --topology all

python tests/glm5_2_mindstudio/configuration_check_benchmark.py \
  --compare --topology all
```

## 7. 官方训练状态监控

Monitor V2 是长时间、低开销的异常筛查，不应让短时 L0/L1 dump capture 跑数百
step。默认仅监控 rank 0 的 `weight_grad`，在 TorchTitan 完成一次 optimizer step
后手动调用一次 `mon.step()`；配置固定 `patch_optimizer_step=false`，避免重复计步。

```bash
# 生成一次共享 fixture
export CUDA_VISIBLE_DEVICES=7
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --data --data-device cuda --topology single

# GPU reference
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture reference --topology single --training-steps 100

# NPU candidate（GPU/NPU 服务器分开时，在 NPU 端同步 fixture 后执行）
unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture candidate --topology single --training-steps 100

# 同步 artifact 后生成索引；不虚构官方不存在的 cross-device verdict
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --compare --topology single --training-steps 100
```

指定一个分布式拓扑或完整矩阵时，四个阶段的 topology 选择必须一致。例如：

```bash
# 代表性分布式拓扑
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture candidate --topology fsdp8 --training-steps 100
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --compare --topology fsdp8 --training-steps 100

# all：分别在 GPU/NPU 端 capture，汇合 artifact 后 compare
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --data --data-device cuda --topology all
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture reference --topology all --training-steps 100

unset CUDA_VISIBLE_DEVICES
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --capture candidate --topology all --training-steps 100
python tests/glm5_2_mindstudio/training_monitor_benchmark.py \
  --compare --topology all --training-steps 100
```

`module`、
`optimizer`、`param`、`cc` 分别用 `--monitor-module`、`--monitor-optimizer`、
`--monitor-param`、`--monitor-cc` 开启；`--monitor-target` 可重复以缩小 module
范围。PP 的 model-parts 命名与 optimizer ownership、Monitor V2 与正式
`torch.compile` 的组合尚需服务器专项验证：当前 PP 会明确警告，显式
`--compile.enable` 会被拒绝，不会静默降级。

capture 的官方 CSV 位于
`mindstudio_artifacts/<experiment-id>/<topology>/<role>-r1/official/rank_<rank>/**.csv`。
`--compare` 只在 `mindstudio_reports/.../<topology>/official_compare/monitor_index.json`
中建立两端 capture 索引；Monitor V2 没有官方 GPU/NPU 数值 comparator，因此这里
不会计算项目自定义 PASS/FAIL。它用于筛出异常 step/rank/module，再用 L0/L1 dump
和正式多 step precision 定位、验收。

## 8. NPU 性能标准流程

当前默认 collector 是 Ascend PyTorch Profiler（`torch_npu_profiler`）。它在
PyTorch 训练进程内按 step schedule 采集 PyTorch、CANN 和 NPU 多层证据，并支持
module、shape、stack、memory 等框架语义。先检查服务器：

```bash
python -m tests.glm5_2_mindstudio.toolchain doctor \
  --scope performance-capture
```

`performance-capture` 对应默认 Ascend PyTorch Profiler 采集机；兼容别名
`performance-torch-npu-capture` 具有相同的采集依赖。离线分析机使用
`--scope performance-analysis`；只有同一环境同时承担采集和分析时才用
`--scope performance`。

先跑不带 profiler 的三次基线；正式数值来自这些 run，而不是 active capture：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
for r in 1 2 3; do
  python tests/glm5_2_mindstudio/performance_benchmark.py \
    --capture --device npu --profiler-off \
    --topology all --replicate "$r"
done
```

单卡、一个分布式拓扑和全部拓扑分别运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=4
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector torch_npu_profiler \
  --topology single --preset overview

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector torch_npu_profiler \
  --topology fsdp8 --preset overview

python tests/glm5_2_mindstudio/performance_benchmark.py \
  --capture --device npu --collector torch_npu_profiler \
  --topology all --preset overview
```

在已有 Ascend PyTorch Profiler 多卡 capture 上运行离线分析并生成 Insight handoff：

```bash
python tests/glm5_2_mindstudio/performance_benchmark.py \
  --analyze --device npu --collector torch_npu_profiler \
  --topology fsdp8 --preset overview --analysis-tools all
```

`msprof-analyze` 的 recipe 不能脱离官方输入矩阵随意混用。默认 Ascend PyTorch
Profiler 产物可进入 offline、advisor、cluster、compare 和 Insight 中各自支持的
流程。显式 `--collector msprof` 仅用于需要命令行包裹整进程的底层或黑盒采集；
其数据可进入 cluster 和 Insight，但不能冒充 advisor/compare 所需的
`*_ascend_pt`。CLI 会在清理旧数据前拒绝 collector/recipe 的无效组合。

性能标准入口当前只允许 NPU；`--device cuda` 保留接口并明确报未实现。完整的采集器
边界、`--type=text`/`db` 容量取舍、compare 和 Insight 阅读方法见
[PERFORMANCE_WORKFLOW_ZH.md](docs/PERFORMANCE_WORKFLOW_ZH.md)。

## 9. 通用参数

所有真实参数以脚本 `--help` 为准。当前主要参数如下：

| 参数 | 含义 | 默认/注意事项 |
| --- | --- | --- |
| `--data` | 生成共享 token plan 与 seed checkpoint | 与 capture/compare 互斥 |
| `--capture reference|candidate` | 运行指定端点 | compile 只允许 candidate |
| `--precheck reference|candidate` | 对已有 L1/mix capture 逐 step、逐 rank 运行 `acc_check`/`multi_acc_check` | 只适用于 migration |
| `--precheck-compare` | 用 `api_precision_compare` 比较两端 details CSV | 要求两端 capture 和 pre-check 都完整 |
| `--graph-visualize` | 对完整 L0/mix migration capture 运行官方分级图比较 | 生成 `.vis.db` 与 TensorBoard 启动索引，不启动服务 |
| `--compare` | 离线读取完整 artifact 并运行/汇总官方比较 | 不启动模型 |
| `--doctor` | 只读检查工具和环境 | 不自动修复 |
| `--doctor-device cuda|npu` | 指定 capture 端的 readiness scope | 未唯一 export 设备时必填 |
| `--list-topologies` | 打印当前拓扑及参数 | 应在正式跑 all 前查看 |
| `--topology NAME` | 单个拓扑或 `all` | 默认 `single` |
| `--topologies a,b` | 显式拓扑集合 | 与 `--topology` 选择语义互斥 |
| `--data-device cuda|npu` | 指定 fixture 生成端 | 也可只 export 一种可见设备变量 |
| `--dump-task statistics|tensor` | 统计量或全量 tensor | 默认 statistics |
| `--level L0|L1|mix` | 模块、API 或两者 | 默认 L0 |
| `--dump-step N` | msProbe 0-based step | 默认 0；训练日志第一个 step 通常是 1 |
| `--dump-steps 0,2` | 多个 0-based dump step | 与 `--dump-step` 互斥 |
| `--dump-ranks 0,3` | 只采指定 global rank | 空配置按官方语义采全部 |
| `--scope VALUE` | 限制官方采集 scope，可重复 | migration capture |
| `--module-or-api VALUE` | 指定官方 module/API list，可重复 | migration capture |
| `--tensor-list VALUE` | 指定 statistics tensor 类别，可重复 | migration capture |
| `--data-mode a,b` | 覆盖 msProbe data_mode | migration capture |
| `--summary-mode statistics|md5` | 统计或校验摘要 | migration capture |
| `--async-dump` | 开启官方异步 dump | 先按目标版本验证完整性 |
| `--no-extra-info` | 关闭官方 extra_info | 会减少调用栈等定位信息 |
| `--backend NAME` | compile checker backend | 默认由 benchmark 配置定义 |
| `--no-dump-graphs` | 不保存 checker graph dump | 只影响 compile workflow |
| `--no-capture-input` | checker 不保存模块输入 | 只影响 compile workflow |
| `--compile-policy glm5-block|children` | block 或子模块标记策略 | 默认 glm5-block |
| `--children-depth N` | children 策略深度 | 只影响 compile workflow |
| `--fuzzy-match` | 官方 compare 模糊匹配 | 映射场景按官方约束使用 |
| `--data-mapping FILE` | 官方 data mapping | 只支持逐 rank compare，且不能与 fuzzy match 同开 |
| `--cell-mapping FILE` | 官方 module/cell mapping | 只支持 L0 module dump |
| `--diff-analysis` | 让官方 compare 识别首差异节点 | compare 阶段 |
| `--tensor-log` | 打印单模块/API tensor 比较日志 | 必须使用 tensor dump |
| `--xlsx` | 让官方 compare 输出 XLSX | 默认保留更轻的 CSV |
| `--precheck-device-ids 0,1` | 指定预检单元测试运行使用的 device | 单 `acc_check` 只接受一个 ID |
| `--precheck-splits N` | 改用 `multi_acc_check`，把 API 单测拆成 N 份 | 当前 CLI 校验范围 1..64；多份 details 不能自动 precheck-compare |
| `--precheck-jit-compile` | 给官方预检开启 JIT compile | dump 与预检的 JIT 语义应按官方说明保持一致 |
| `--precheck-save-error-data` | 保存未达标 API 的错误数据 | 体积和敏感性显著增加 |
| `--precheck-filter-api` | 启用官方 API 过滤 | 具体黑白名单以锁定版本为准 |
| `--precheck-config FILE` | 传入官方预检 YAML 配置 | 文件内容会进入 operation identity |
| `--precheck-resume-csv FILE` | 从官方 `accuracy_checking_result_*.csv` 续检 | 只允许一个 topology/step/rank；同目录 details 必须存在 |
| `--training-steps N` | 覆盖 Monitor V2 训练步数 | 只允许 monitor workflow |
| `--monitor-ranks 0,1` | 选择 Monitor V2 global ranks | 默认脚本为 rank 0；空配置表示全部 |
| `--monitor-start-step/--monitor-stop-step` | inclusive 起点与 exclusive 终点 | stop 未设置时由 collect_times 限制 |
| `--monitor-step-interval N` | 每 N step 写一次 | 默认 1 |
| `--monitor-record-steps N` | 每个 CSV 聚合的 step 数 | 默认脚本 10 |
| `--monitor-collect-times N` | 最大写出次数 | 默认脚本 100 |
| `--monitor-ops a,b` | CSV 统计列 | 支持 min/max/mean/norm/nans |
| `--monitor-module/optimizer/param/cc` | 增量开启对应官方监控模块 | weight_grad 默认开启；额外模块会增加开销 |
| `--monitor-target TEXT` | module 名字包含过滤 | 可重复 |
| `--graph-overflow-check` | 标注分级图溢出/下溢 | graph_visualize only |
| `--graph-progress-log` | 打印官方构图进度 | graph_visualize only |
| `--collector NAME` | 性能采集入口 | 默认 `torch_npu_profiler`；底层或黑盒整进程采集可显式选择 `msprof` |
| `--collector-arg=ARG` | 追加当前 CANN 版本确认过的 msProf 参数 | 可重复；不能覆盖 lifecycle 管理的 output/application/dynamic |
| `--preset NAME` | Ascend PyTorch Profiler 策略 | msProf 保持 `overview` 占位，不接受 `all` |
| `--analysis-tools none|offline|advisor|cluster|all` | 性能离线分析阶段 | msProf 的 `all` 只选 cluster；torch_npu 的 `all` 选 offline+advisor+cluster |
| `--cluster-mode all|communication_time|communication_matrix` | 官方 cluster 解析范围 | 默认 `all`；只定位耗时或矩阵时可缩小 |
| `--cluster-agent-output` | 请求官方 cluster JSON stdout | stdout 连同命令、stderr、return code 保存到 `cluster.json` |
| `--cluster-bypass-input-safety-checks` | 传递官方 cluster `--force` | 绕过属主、权限和超大文件检查；与实验生命周期 `--force` 无关 |
| `--compare-baseline PATH` | msprof-analyze compare 基线 | candidate 必须是 torch_npu `*_ascend_pt`；支持满足官方格式的 NPU/NPU 或 GPU/NPU 比较 |
| `--repeat N` | 选择 accuracy endpoint 已声明的重复编号 | migration/compile/config-check/monitor，默认脚本声明 1 次 |
| `--replicate N` | 性能独立运行编号 | performance profiler-off/active A/B；不同编号进入不同目录 |
| `--force` | 删除所选旧 generation/阶段后重跑 | 不能和旧数据混搭 |
| `--dry-run` | 只打印所选阶段计划，不启动训练或写入/清理 generation | 用于审查命令；不能证明工具可用 |

## 10. force、继续运行和跨服务器同步

- `--force --data`：清除该实验旧 fixture、run、artifact、report，建立新 generation；
- `--force --capture ROLE`：只清除所选 role/topology 的 run、artifact、report；
- 不加 `--force`：跳过 config hash、fixture generation 和附件校验均完整的 capture；
- 不完整/失败目录：先归档，再重跑该成员；
- 活跃 PID：拒绝删除或复用，防止杀掉仍在运行的任务；
- compare：只读完整 artifact；同一实验的 fixture generation 不一致时必须拒绝。

跨服务器最少同步：

```text
mindstudio_fixtures/<experiment-id>/
mindstudio_artifacts/<experiment-id>/<topology>/reference-r1/
mindstudio_artifacts/<experiment-id>/<topology>/candidate-r1/
```

若执行 API pre-check，还必须同步 endpoint artifact 下的
`<topology>/precision_precheck/reference-r1/` 与 `candidate-r1/`。若只查看分级图，
则同步 `<topology>/graph-visualize-r1/` 的处理后数据库和 tracked report 即可；重新
生成图仍需完整 capture artifact。

`mindstudio_runs/` 是运行日志和输入契约，适合排错但不是离线 compare 的主要输入。
raw tensor 可能包含敏感数据，不应直接提交 Git 或公开 Release。

`mindstudio_fixtures/`、`mindstudio_runs/`、`mindstudio_artifacts/` 默认被 Git
忽略；`mindstudio_reports/` 保持可审阅、可选择提交。跨服务器可直接用 `rsync`
保留相对目录：

```bash
rsync -a --partial \
  user@gpu-host:/workspace/torchtitan-test/mindstudio_fixtures/<experiment-id>/ \
  mindstudio_fixtures/<experiment-id>/

rsync -a --partial \
  user@gpu-host:/workspace/torchtitan-test/mindstudio_artifacts/<experiment-id>/ \
  mindstudio_artifacts/<experiment-id>/
```

也可以使用仓库的 Release 传输工具。CLI 的兼容默认仍是 `full`，它包含 fixture、
run、raw dump/checkpoint 等无损数据，体积大且敏感；日常同步报告给本地/Codex 分析
应显式选择推荐的 `--content analysis`。analysis 保留报告、紧凑制品、日志、CSV/
JSON/HTML 以及处理后的 `.vis.db`，但排除 raw tensor、checkpoint、CANN 原始采集和
编译 cache：

```bash
python release_artifacts.py upload <experiment-id> --content analysis
# 只有确实需要无损续跑/重新解析时才用：
python release_artifacts.py upload <experiment-id> --content full
python release_artifacts.py download <experiment-id> --overwrite
```

`.vis.db` 已经是 graph_visualize 的处理后数据库，不等于 raw tensor；但其中仍可能
包含模块名、统计量、源码与服务器路径。两种 content 模式上传前都必须审查，full
还必须确认 tensor、token、checkpoint 是否允许离开服务器。

## 11. 输出目录与阅读顺序

```text
mindstudio_fixtures/<experiment-id>/
mindstudio_runs/<experiment-id>/<topology>/<role>-r1/
mindstudio_artifacts/<experiment-id>/<topology>/<role>-r1/
└── official/                 # msProbe 原始输出
mindstudio_artifacts/<experiment-id>/<topology>/precision_precheck/<role>-r1/
└── stepN/rankN/official/     # 单端 acc_check/multi_acc_check
mindstudio_reports/<experiment-id>/<topology>/
├── official_compare/        # 官方 dump compare 或 compile CSV 汇总
├── precision_precheck/compare-r1/ # api_precision_compare + HTML/JSON 索引
└── graph-visualize-r1/      # tracked .vis.db 索引和 TensorBoard 命令
```

每个 subprocess 的终端都会打印 `Runtime log:`。建议按以下顺序阅读：

1. `mindstudio_reports/.../README.md` 或 `<experiment-name>.html`：实验契约、拓扑、状态；
2. `official_summary.json`：项目对官方文件的紧凑索引，不替代原文件；
3. 官方 CSV/XLSX：逐模块/API 的 `Result`、`Err_Message` 和误差指标；
4. capture `manifest.json`：fixture generation、工具版本、官方文件 hash；
5. `mindstudio_runs/.../runtime.log`：完整命令与异常栈；
6. 若官方结果异常，按本目录的官方诊断路线缩小 step、rank、module、API 或算子范围后重新采集。

详见 [OUTPUTS_AND_REPORTS_ZH.md](docs/OUTPUTS_AND_REPORTS_ZH.md)。

## 12. 不应混淆的结论

| 证据 | 能回答 | 不能单独回答 |
| --- | --- | --- |
| msProbe config check | 两端影响精度的配置差异 | 张量是否对齐 |
| msProbe API pre-check | 单端 API 相对 CPU 标杆是否异常，以及两端预检结论是否一致 | 完整模型端到端是否收敛 |
| msProbe L0/L1 compare | 首个可疑模块/API、误差量级 | 5000 step 是否稳定收敛 |
| PrecisionChecker single-pass | 被标记模块 eager/compile 的前反向差异 | 完整 fullgraph 训练是否正确 |
| TrainerMonitorV2 | 长时训练中的激活、梯度、权重和优化器状态趋势 | 两端逐 tensor 的最终比较 |
| msProf / Ascend PyTorch Profiler | 框架、CANN、Device、通信和内存原始性能证据 | 数值精度是否通过 |
| msprof-analyze advisor | 基于规则的候选瓶颈和调优建议 | 不经原始证据验证的最终根因 |
| msprof-analyze cluster | rank/stage 时间、通信域、通信矩阵和慢链路 | 单算子实现为何变慢 |
| msprof-analyze compare | 两份 profile 的计算、通信、调度、算子和内存差异 | 模型输出是否数值一致 |
| MindStudio Insight | Summary、Timeline、Communication、Operator、Memory 的交互式分析 | 自动替代工程师作结论 |

标准迁移结论应是多条证据的交集，而不是选一张表宣布完成。
