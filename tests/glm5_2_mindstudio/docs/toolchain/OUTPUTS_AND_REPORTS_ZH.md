# MindStudio 官方产物、紧凑制品、中文报告与同步规范

> 状态：当前布局与长期产物规范。
>
> 本文先准确描述当前已实现目录，再说明长期同步/保留规范。官方工具的内部文件名可能随版本变化，项目层通过 manifest 记录实际文件，不能假设固定目录结构后静默丢文件。

## 1. 为什么必须分三层

一次标准实验在逻辑上会同时产生三种性质完全不同的数据：

1. **官方原始产物**：用于重新比较、深度定位或导入官方工具；体积可能很大；
2. **可校验 artifact/紧凑选择**：用于跨服务器同步、完整性校验、自动报告和 GitHub Release；
3. **中文索引/报告**：用于理解实验、链接官方结果和项目补充结果；不能代替原始证据。

```text
训练/capture
  → artifact/official（官方 raw，保真、较大、受控保存）
  → manifest + hashes + complete marker
  → Release analysis 选择（可选的紧凑同步集）
  → official compare/analyze
  → 中文 <experiment>.html / README.md（解释与导航）
```

如果只同步 HTML，会丢失重新分析所需的数据；如果把全部 raw 都同步或提交 Git，又会造成数百 GB 存储、路径长度和敏感数据问题。分层是必须的。

当前实现没有再复制一份 `official_raw` 到 run，也没有固定生成一个独立
`compact_artifact/` 目录。官方 raw 位于 artifact 的 `official/` 下；
`release_artifacts.py --content analysis` 在传输时选择轻量结果。不要把本文的逻辑
分层误解成三份物理副本。

## 2. 当前输出根目录

当前已实现并延续现有 fixture/run/artifact/report 语义：

```text
mindstudio_fixtures/accuracy/     # 精度流程跨端共享的固定输入
mindstudio_runs/                  # 原始运行根目录，按 accuracy/performance 分族
├── accuracy/                     # msProbe/compile/monitor
└── performance/                  # system/operator/memory
mindstudio_artifacts/accuracy/    # 精度官方 raw + manifest/hash/complete
mindstudio_artifacts/performance/system/ # 系统性能轻量状态
mindstudio_reports/accuracy/      # 精度 compare、摘要与中文入口
mindstudio_artifacts/accuracy/<experiment-id>/diagnoses/<case-id>/case.json # 诊断控制面
mindstudio_reports/performance/system/ # 系统性能报告
```

诊断 case 不复制 official raw。每个 case 只保存 `case.json`、下一步 recipe、
人类可读报告和指向 fixture/run/artifact/report 的证据路径。该目录默认忽略，是否
同步由问题敏感性和协作方式决定；它不能替代被引用的原始产物。

每次新的精度 capture 在自己的 run 根目录额外保留
`training_metrics.jsonl`，用于整网 Loss/Grad Norm 现象分析。Case 下的
`03_observe/` 只包含由该文件生成的 CSV、JSON、交互 HTML 和 SVG 派生证据；msProbe 的 dump、
Monitor CSV、`.trend.db` 和 `.vis.db` 仍留在各自官方 artifact/report 目录。

实验 ID 应简洁但能区分工作流、设备、模式、拓扑和关键训练配置。例如：

```text
migration-cuda-npu-bf16-random-s1-b64-seq128-seed61-a1b2c3d4
compile-npu-inductor-bf16-random-s1-b64-seq128-seed61-e5f6a7b8
```

命名原则：

- 所在根目录已经表达的信息不重复；
- `cuda`/`npu` 不再写成 `gpu-cuda`/`npu-npu`；
- role 子目录只写 `reference-r1`、`candidate-r1`，不重复 `candidate-npu-candidate`；
- 末尾短 hash 表示完整实验契约，不为每个阶段各算一套不相干 hash；
- Windows 路径长度敏感，避免把所有参数展开成超长目录名；完整参数保存在 manifest。

## 3. 精度实验的逻辑层级

`<experiment-id>` 是一项完整精度实验，不是一次命令。其下依次按以下层级组织：

```text
<experiment-id>                         # 模型、设备对、dtype、输入契约等整体身份
└── <topology>                          # single、fsdp8、tp8、pp8、hsdp2x4 等
    └── <stage-family>/<stage>           # 检查、正常观察、采集或按现象诊断
        └── <stage-profile>              # 本阶段的步数、范围和配置 hash
            └── <role>-r<repeat>         # GPU reference / NPU candidate / 重复编号
```

所以 `single` 下出现多个一级分类是预期行为：它们不是重复实验，而是同一 topology
的标准精度流程。

| 目录 | 作用 | 什么时候产生 |
| --- | --- | --- |
| `checklist/configuration-check` | 排除超参、版本、模型结构、权重、输入和 rank 映射不一致 | 标准流程第一阶段 |
| `observations/training/<profile>` | 无 PrecisionDebugger、无 Monitor hook 的正常训练；观察 Loss、Grad Norm、NaN/Inf、首步差异、长稳漂移和尖刺 | CheckList 之后，细粒度 dump 之前 |
| `observations/monitor/<profile>` | 使用 Monitor V2 低开销监控所选参数、梯度、模块或通信统计量 | 长程异常存在但第一现场 step/module 不明确时 |
| `captures/<dump-profile>` | 使用 PrecisionDebugger 采集指定 step/rank 的 statistics/tensor/md5 数据 | 已确定需要细粒度比较的位置时 |
| `diagnostics/dump/<dump-profile>` | 在既有实验内追加不同 scope、level、task 或 step 的定点 dump | 对首个可疑模块/API 继续缩小范围时 |
| `diagnostics/monitor/<profile>` | 在既有实验内追加特定窗口或对象的 Monitor 诊断 | baseline/初次 monitor 后继续定点监控时 |

早期实现把这一阶段称为 `baseline`，但该词容易同时被误解为 GPU 标杆、性能基线，
或与一个并不存在的 `non-baseline` stage 成对。当前用户接口统一称为
`observation`，目录称为 `observations/training`。它表示“无精度工具 hook 的正常训练
观察”，与 `reference/candidate` 设备角色是两个正交维度。同一个 observation profile
内同时有 `reference-r1`（GPU）和 `candidate-r1`（NPU），两端运行相同 checkpoint、
token plan 和训练契约，再比较逐 step 训练现象。它回答：

```text
整网是否已经出现问题？
问题属于 NaN/Inf、首 Step、后续漂移还是尖刺？
最早应当采集哪个 step？
```

它不回答具体哪个 API 出错；后者由 Monitor、PrecisionDebugger、`msprobe compare`
和分级可视化完成。因此标准流程不是一开始就 dump。

## 4. 当前完整目录树与文件含义

以下树使用 `<profile>` 表示配置摘要，例如 baseline 的 `s500-fb08c375`；完整配置
永远保存在 JSON、manifest 和启动契约中，不依赖人工反解 hash。

### 4.1 固定输入：`mindstudio_fixtures`

```text
mindstudio_fixtures/accuracy/<experiment-id>/<topology>/
└── inputs/<input-profile>/
    ├── fixture.json                 # generation ID、配置、文件 hash 和完整性状态
    ├── checkpoint/                  # GPU/NPU 共用的初始化权重
    ├── token_plan.*                 # 每个训练 step 的固定 token/sample 计划
    ├── token_generation.log         # token plan 生成日志
    └── seed_generation.log          # checkpoint/随机初始化生成日志（如适用）
```

fixture 是所有阶段的共同输入契约，不是 msProbe dump。`<topology>` 必须保留，因为
DP/TP/PP/CP 等切分会改变 rank 消费输入的方式。

### 4.2 运行现场：`mindstudio_runs`

```text
mindstudio_runs/accuracy/<experiment-id>/<topology>/
├── checklist/
│   └── configuration-check/
│       ├── reference-r1/            # GPU ConfigChecker 采集现场
│       └── candidate-r1/            # NPU ConfigChecker 采集现场
├── observations/
│   ├── training/<profile>/
│   │   ├── reference-r1/            # GPU 无 hook 正常训练
│   │   └── candidate-r1/            # NPU 无 hook 正常训练
│   └── monitor/<profile>/
│       ├── reference-r1/            # GPU Monitor V2（工具支持时）
│       └── candidate-r1/            # NPU Monitor V2
├── captures/
│   └── <dump-profile>/
│       ├── reference-r1/            # GPU PrecisionDebugger 采集现场
│       └── candidate-r1/            # NPU PrecisionDebugger 采集现场
└── diagnostics/
    ├── dump/<dump-profile>/<role>-rN/
    └── monitor/<profile>/<role>-rN/
```

每个 `<role>-rN/` 运行目录使用同一套外围文件：

```text
<role>-rN/
├── EXPERIMENT.md                    # 人进入目录后首先阅读的实验摘要与原始入口命令
├── runtime.log                      # 完整实时训练/工具日志
├── run_state.json                   # PID、attempt、running/completed/failed/interrupted
├── resolved_command.json            # 最终 argv 数组
├── resolved_launch.sh               # 最终环境变量与可复现启动命令
├── msprobe_config.json              # 当前阶段工具配置；baseline 明确为 instrumentation=none
├── training_metrics.jsonl           # 逐 step TorchTitan 指标；baseline 必需，其他训练也保留
├── input_contract/
│   ├── rank-*.jsonl                 # 每个 rank 实际消费的 step/sample/token 证据
│   └── summary.json                 # 对 token plan、DP replica 和 step 连续性的校验摘要
└── trainer_output/                  # TorchTitan 自身输出（仅配置要求时出现）
```

并非每个 stage 都会填满所有可选文件。例如 config-check 主要产生 ZIP，baseline 不
启动 msProbe，dump 才会在 artifact 中产生 `dump.json`/`construct.json`。

### 4.3 可同步端点证据：`mindstudio_artifacts`

artifact 路径镜像 run 的 stage/profile/role 路径：

```text
mindstudio_artifacts/accuracy/<experiment-id>/<topology>/
├── checklist/configuration-check/<role>-rN/
├── observations/training/<profile>/<role>-rN/
├── observations/monitor/<profile>/<role>-rN/
├── captures/<dump-profile>/<role>-rN/
└── diagnostics/{dump|monitor}/<profile>/<role>-rN/
```

每个 endpoint artifact 的外围结构一致：

```text
<role>-rN/
├── official/                        # 该阶段交给下游分析的规范化端点输出
├── manifest.json                    # fixture、topology、输入契约、工具链及逐文件 hash
└── complete.json                    # 通过结构/输入校验后才生成的完成标记
```

`official/` 是历史通用名称，含义是“该 endpoint 的规范化输出”，不保证一定由
msProbe 生成。各阶段内容如下：

```text
checklist/.../official/
└── config_check_rankN.zip           # 官方 ConfigChecker 每 rank 数据包

observations/training/.../official/
└── training_metrics.jsonl           # 从 run 选择出的正式逐 step 训练指标

observations/monitor/.../official/
└── rank_N/**/*.csv                  # Monitor V2 按 rank/对象输出的时间序列

captures/<dump-profile>/.../official/
└── stepN/rankN/
    ├── dump.json                    # Module/API 输入输出统计或 tensor 索引
    ├── construct.json               # Module/API 层级结构
    ├── stack.json                   # API 调用栈（由配置和版本决定）
    └── dump_tensor_data/            # task=tensor 时的完整 tensor（如产生）
```

单端精度预检是 dump artifact 的派生端点证据，不写回原 capture：

```text
mindstudio_artifacts/accuracy/<experiment-id>/<topology>/
└── precision_precheck/
    ├── reference-rN/stepN/rankN/
    │   ├── official/                # GPU API-vs-CPU 高精度预检 CSV
    │   ├── invocation.json
    │   ├── runtime.log
    │   ├── manifest.json
    │   └── complete.json
    └── candidate-rN/stepN/rankN/    # NPU API-vs-CPU 高精度预检，结构同上
```

### 4.4 汇合分析与人类入口：`mindstudio_reports`

```text
mindstudio_reports/accuracy/<experiment-id>/
├── <experiment-id>.html             # 当前 workflow 的自包含聚合报告
├── README.md                        # Markdown 入口
├── report.json                      # 机器可读聚合数据
└── <topology>/
    ├── checklist/configuration-check/
    │   └── official_compare/        # 官方 config_check 逐 rank 比较
    ├── observations/training/<profile>/
    │   └── comparison/              # 项目实现的整网训练现象比较，不是 msprobe compare
    ├── observations/monitor/<profile>/
    │   └── official_compare/        # 只索引两端 Monitor；不伪造跨端 PASS/FAIL
    ├── captures/<dump-profile>/
    │   ├── official_compare/        # 官方 msprobe compare CSV/JSON/XLSX
    │   ├── precision_precheck/
    │   │   └── compare-rN/          # 官方两端 API 预检比较及 HTML 索引
    │   ├── trend-<role>-rN/         # 官方 data2db 生成的 .trend.db
    │   └── overflow-<role>-rN/      # 官方首溢出分析
    └── diagnostics/{dump|monitor}/<profile>/
        └── 对应的 compare/trend/overflow 派生结果
```

baseline 的 `comparison/` 内容全部由项目基于官方诊断流程生成：

```text
comparison/
├── training_metrics_compare.csv     # GPU/NPU 按 step 对齐后的 Loss/Grad Norm
├── summary.json                     # NaN/Inf、首步、长稳和尖刺现象分类
├── loss.svg                         # 两端 Loss 曲线
├── training_observation.html         # Panel + ECharts 自包含交互面板，可悬停、缩放、平移、查看数据和导图
├── grad_norm.svg                    # 两端 Grad Norm 曲线
├── loss_relative_error.svg          # Loss 误差、零误差 baseline 与 1% 指导线
├── grad_norm_relative_error.svg      # Grad Norm 误差及零误差 baseline
├── loss_signed_difference.svg        # Loss 的 NPU-GPU 有符号差值及零差值线
├── grad_norm_signed_difference.svg   # Grad Norm 的 NPU-GPU 有符号差值及零差值线
├── early_loss.svg                    # 首 10 个已观测 step 的 Loss 双线放大图
├── early_loss_relative_error.svg     # 首 10 个 step 的 Loss 相对误差放大图
├── grad_norm_signed_relative_error.svg # Grad Norm 有符号相对误差及 ±5% 指导线
├── official_summary.json            # 生命周期摘要；不表示 msProbe verdict
├── runtime.log                      # 比较过程日志
├── toolchain_compatibility_diff.json# 两端来源差异，仅供人工检查
├── comparison_state.json            # 输入 identity/hash 与输出索引，用于安全复用
└── compare_run_state.json            # compare attempt/PID/完成状态
```

名字 `official_compare` 只保留给确实调用官方比较器，或直接索引官方输出的阶段。
baseline 已使用 `comparison/`，因为 CSV、SVG、summary 均由项目生成。

其他官方派生分析：

```text
mindstudio_runs/accuracy/<experiment-id>/<topology>/graph-visualize-rN/
├── invocation.json
└── runtime.log

mindstudio_artifacts/accuracy/<experiment-id>/<topology>/graph-visualize-rN/
├── official/*.vis.db                # 官方分级构图数据库
├── manifest.json
└── complete.json

mindstudio_reports/accuracy/<experiment-id>/<topology>/graph-visualize-rN/
├── README.md                         # TensorBoard 启动说明
└── index.json                        # 数据库相对路径、大小、hash 和来源
```

compile accuracy 是独立实验族，而非统一 eager 诊断四阶段之一：

```text
mindstudio_runs/accuracy/<compile-experiment>/<topology>/candidate-rN/
mindstudio_artifacts/accuracy/<compile-experiment>/<topology>/candidate-rN/
└── official/
    ├── precision_rankN_partM.csv     # 官方 PrecisionChecker 原始分片结果
    ├── precision_rankN.csv           # 项目聚合后的逐 rank 结果
    └── compile_coverage_rankN.json   # 前向/反向和目标模块覆盖证据
mindstudio_reports/accuracy/<compile-experiment>/<topology>/official_compare/
└── compile_all_ranks.csv             # 跨 rank 汇总；判定仍来自官方行
```

若官方工具只能在指定相对结构中工作，`official/` 内完整保留该结构；项目层不把
其文件全部扁平化重命名。目录树是导航契约，实际文件清单和 SHA-256 仍以
`manifest.json` 为准。

## 5. 各阶段应该保存什么

### 5.1 fixture

fixture 是 reference/candidate 的共同契约，至少包含：

- 固定 token plan 或可确定复现输入；
- seed checkpoint；
- module/config、dtype、step、batch/token、sequence length、seed；
- topology 约束和数据分发契约；
- 每个文件的大小与 SHA-256；
- generation ID、config hash、创建状态；
- token/checkpoint 生成命令和日志。

fixture 不等于 msProbe dump。msProbe capture 必须引用 fixture，而不是自己生成另一份随机数据。

### 5.2 capture run

每个 reference/candidate run 保存：

- 完整命令，使用 argv 数组保存而不是不可逆字符串；
- 环境白名单（设备可见性、CANN 路径摘要、关键开关）；
- `runtime.log`；
- 官方配置文件；
- 官方 raw 目录；
- 三仓与官方工具的 commit/version；
- rank/device/topology/step/range/module 选择；
- 开始/结束时间、PID/进程组、exit code；
- fixture/config/toolchain hash；
- 完成标记与附件清单。

训练结束后，workflow 会验证 `input_contract/rank-*.jsonl`：每个 local rank 的
optimizer step 数和连续顺序、推导出的 DP/PP rank、每 step 的 global slots、
sample SHA-256 与共享 token plan，以及同一 DP replica 的输入一致性都必须成立。
校验摘要写入 `input_contract/summary.json` 和 artifact manifest；
`input_contract.valid=true` 是发布 `complete.json` 的必要条件。

compile workflow 还要求 `official/compile_coverage_rank<rank>.json` 存在：每个
rank/model-part 必须有非零 GLM block，并且每个 part CSV 至少有非 SKIP 的模块
forward 与 backward 决定性行。CSV 表头或合成 `LOSS` 行本身不能表示 checker
真正执行过目标模块。

完成标记只能在以下条件同时满足时写入：

1. 子进程 exit code 成功；
2. 官方预期根目录存在；
3. 关键产物通过最小结构校验；
4. 运行时输入契约通过并写入 manifest；
5. manifest 和附件 hash 已写完；
6. 没有临时文件仍在写入。

“目录存在”“runtime.log 存在”都不足以表示 capture 完成。

### 5.3 offline compare/analyze

保存：

- compare 命令和官方配置；
- reference/candidate artifact identity；
- compare 工具版本和 source commit；
- compare runtime log；
- 官方 CSV/JSON/HTML/XLSX/数据库等实际输出；
- 项目提取的轻量 summary；
- 无法解析字段的警告，禁止静默忽略。

compare 应可重复执行，不修改 reference/candidate raw。官方输出 schema 变化时，原始 compare 结果仍可人工阅读，项目 parser 失败不能删除它。

### 5.4 precision pre-check

单端预检结果属于 endpoint artifact，但不会写回原 capture 的 `<role>-rN/`；它放在
同一 topology 下独立的 `precision_precheck/<role>-rN/`。两端比较结果才属于
report。当前每个 step/rank 独立保存：

- 单端 `acc_check`/`multi_acc_check` 的 result/details CSV；
- 两端 `api_precision_compare` 的 result/details CSV；
- 完整 invocation、runtime log、operation digest；
- 输入 artifact manifest、complete marker 和 `dump.json` 的 SHA-256；
- `manifest.json` 与 `complete.json`。

`compare-r1/precheck_report.html` 是逐 step/rank 的人类可读入口，
`precheck_index.json` 是机器可读索引；两者只链接并汇总官方 CSV，不重新定义官方
判定。

因此跨服务器执行预检比较时，除了两端 capture artifact，还必须同步：

```text
mindstudio_artifacts/<experiment-id>/<topology>/precision_precheck/reference-r1/
mindstudio_artifacts/<experiment-id>/<topology>/precision_precheck/candidate-r1/
```

项目会校验预检结果对应当前 capture；旧 details CSV 不能与新 dump 混搭。

## 6. 官方原始产物的分类

### 6.1 msProbe

不同版本、task 和 dump 配置会产生不同文件。当前项目只规定 artifact 容器结构，
并在 manifest 中索引官方实际文件，不再创造一层虚构的 `official_raw/`：

```text
mindstudio_artifacts/accuracy/<experiment-id>/<topology>/captures/<dump-profile>/<role>-r1/
├── official/
│   └── stepN/rankN/
│       ├── dump.json
│       ├── stack.json或锁定版本实际输出
│       └── dump_tensor_data/（tensor task 才可能存在）
├── manifest.json
└── complete.json
```

普通离线 compare 放在同一 dump profile report 的 `official_compare/`；单端预检放在 endpoint
artifact 的 `precision_precheck/<role>-rN/.../official/`；两端
`api_precision_compare` 才放在 report 的 `precision_precheck/compare-rN/`。这些
目录可能包含 CSV、JSON、HTML、XLSX 或其他官方格式，以锁定版本实际产出为准。

### 6.2 Monitor V2 与 graph_visualize

Monitor capture 沿用普通 endpoint artifact：

```text
mindstudio_artifacts/accuracy/<experiment-id>/<topology>/observations/monitor/<profile>/<role>-r1/official/
└── rank_<rank>/**/*.csv

mindstudio_reports/accuracy/<experiment-id>/<topology>/observations/monitor/<profile>/official_compare/
├── monitor_index.json
├── official_summary.json
└── runtime.log
```

`monitor_index.json` 只把 GPU/NPU capture 身份连在一起。Monitor V2 没有官方的
cross-device 数值 comparator，因此这里不生成项目自定义 PASS/FAIL；CSV 用于发现
异常 step/rank/module，再转入 L0/L1 dump 与多 step precision。

分级图把处理后数据库与 tracked 索引分开：

```text
mindstudio_runs/<experiment-id>/<topology>/graph-visualize-r1/runtime.log
mindstudio_artifacts/<experiment-id>/<topology>/graph-visualize-r1/official/*.vis.db
mindstudio_reports/<experiment-id>/<topology>/graph-visualize-r1/
├── README.md
└── index.json
```

`.vis.db` 是官方 `graph_visualize` 已处理的层级比较数据库，不是 raw tensor；
`index.json` 保存相对路径、大小、SHA-256 和 TensorBoard 命令。数据库仍可能包含
模块名、统计量、源码/服务器路径，因此同步前必须审查。

必须保留：

- 数据级别（统计量或 tensor）；
- forward/backward/API/module 范围；
- rank 和 step 范围；
- dump 配置；
- 官方版本；
- 文件 hash。

### 6.3 Ascend PyTorch Profiler/CANN

性能 raw 可能包含：

- CANN profiling 原始目录；
- framework trace；
- timeline/trace JSON；
- profiler 数据库；
- kernel、通信、内存、op summary；
- system collector 数据；
- 多 rank 目录。

这些数据是体积最大的类别。现有 performance workflow 已管理它们，MindStudio 标准目录应通过 manifest/link 引用，而不是复制一份。

### 6.4 msprof-analyze

根据 recipe/version，结果可能包括：

- advisor HTML/XLSX；
- cluster/communication 分析；
- 算子、调度、融合或瓶颈统计；
- JSON/CSV/Excel；
- 分析日志。

报告应显示“哪个 recipe 成功”“输出在哪里”，不能把 advisor 阶段成功误写成模型性能 PASS。

### 6.5 MindStudio Insight

Insight 是查看器。项目应保存它能够导入的数据和导入说明；如人工从 GUI 导出截图、标注或工程文件，可放到 report `assets/insight/`，并记录：

- Insight 版本；
- 导入的数据 ID/hash；
- rank、时间范围和筛选条件；
- 截图/结论作者与时间。

截图是解释材料，不是原始证据。

### 6.6 图编译

图模式现有产物包括但不限于：

- `TORCH_TRACE`/tlparse 数据；
- Dynamo explain、graph break、guard、recompile 日志；
- FX Graph；
- AOTAutograd/Inductor IR；
- 生成代码；
- backend 日志和 fallback 信息；
- eager/graph 精度和性能结果。

MindStudio compile accuracy 的官方输出作为独立子目录加入，不覆盖这些编译调试证据。

## 7. compact artifact 的定义

当前 workflow 不自动生成一个名为 `compact artifact` 的独立目录。跨
GPU/NPU/CPU 做官方 compare 时，必须同步完整 `mindstudio_artifacts/.../<role>-rN`
（其中包含 official raw）。`release_artifacts.py --content analysis` 才会在发布时
选择轻量分析集合；这种集合用于阅读，不保证还能重新执行 tensor compare。

长期保留或 Release 的 compact 选择建议包含：

- manifest 和工具链元信息；
- 官方配置文件；
- runtime log（必要时脱敏/截取，但原始受控保存）；
- 官方 compare 的轻量结果；
- 项目 summary JSON；
- 选定的小型 trace/CSV/HTML；
- raw 数据位置、大小、hash 清单；
- raw 未同步时的明确 `available=false`，而不是制造坏链接。

不默认包含：

- 全量 tensor dump；
- checkpoint 大分片；
- CANN 全量 raw；
- 多 rank 重复数据库；
- compiler cache；
- TensorBoard 大事件文件；
- Python/Conda 环境副本；
- 可从 raw 确定性重建的大型中间文件。

## 8. 当前 manifest 核心字段

```json
{
  "schema": "torchtitan.glm5_2.mindstudio_artifact",
  "schema_version": 1,
  "workflow": "migration",
  "role": "candidate",
  "repeat": 1,
  "experiment_digest": "...",
  "fixture_generation_id": "...",
  "fixture_checkpoint_sha256": "...",
  "fixture_token_plan": {},
  "input_contract": {
    "valid": true,
    "mapping": "optimizer-step-global-slot-v1",
    "validated_global_ranks": [0]
  },
  "topology": {"name": "single", "world_size": 1},
  "toolchain": {},
  "official_output": "official",
  "official_files": [],
  "runtime_log": "/diagnostic/absolute/path/runtime.log"
}
```

`complete.json` 另存 status、experiment digest、fixture generation 和原始 attempt
ID。`official_files` 对 official raw 逐文件记录相对路径、大小和 SHA-256。

路径约束：

- official 文件索引使用 artifact 内相对路径；
- 绝对服务器路径只作为提示，不能成为离线 compare 的唯一定位方式；
- Windows/Linux 分隔符统一归一化到 manifest 的 `/`；
- 每个 attachment 逐文件校验；大型 raw 可用分层索引 hash，避免每次 compare 重扫数百万文件。

## 9. 中文索引应该包含什么

`mindstudio_reports/<experiment-id>/<experiment-id>.html` 和同目录 `README.md`
是入口，不是另一套官方 analyzer。当前报告重点索引 workflow、topology、状态、
官方 summary、runtime log 和分析入口；报告应包含：

1. **实验目的**：迁移、编译精度、性能还是联合定位；
2. **实验契约**：模型、dtype、step、token/batch、seq、seed、fixture；
3. **端点与拓扑**：reference/candidate、设备、world size、并行配置；
4. **环境**：三仓 commit、torch/torch_npu、CANN、官方工具版本；
5. **阶段状态**：data/capture/compare/analyze/report 是否完成；
6. **官方精度结果**：链接 msProbe 原始比较文件和紧凑摘要；
7. **训练现象**：loss、grad norm、NaN/Inf、尖刺和重复运行状态；
8. **精度定位**：首个异常 module/API、预检和 graph_visualize；
9. **编译证据**：PrecisionChecker 前反向、graph、warning/skip；
10. **性能证据**：msProf/Ascend PyTorch Profiler、advisor、cluster、compare、Insight 导入根；
11. **结论与下一步**：人工根因、缩小采集范围、修复、A/B、未决问题；
12. **产物清单**：raw/compact/report 的位置、大小、hash 和可用性。

状态颜色只能表达阶段：

- `COMPLETE`：阶段完成且产物校验通过；
- `PARTIAL`：存在可保留结果但阶段未完成；
- `FAILED`：命令失败或关键产物缺失；
- `SKIPPED`：配置未要求该阶段；
- `UNAVAILABLE`：外部工具/原始数据未同步。

不能把 `COMPLETE` 写成精度 `PASS`。精度结论应来自官方比较和明确标准。

## 10. 跨服务器同步流程

### 10.1 fixture

fixture 可在 GPU 或 NPU 任一端生成，但只能生成一次。同步后两端先校验 manifest/hash，再 capture。

```text
fixture creator
  → fixture manifest + token plan + checkpoint
  → reference server
  → candidate server
```

### 10.2 capture

GPU/NPU 各自运行自己的模型代码和官方采集。完成后同步完整 artifact 到 compare 机：

```text
GPU capture → reference artifact ┐
                                 ├→ CPU/NPU compare host
NPU capture → candidate artifact ┘
```

statistics compare 也需要 `dump.json` 等 official raw，tensor compare 还需要
`dump_tensor_data`。最稳妥的做法是同步完整 role artifact。不要假设 Git 会同步
这些被忽略目录；同步工具必须按相对目录复制并在 compare 前让 manifest/hash
重新校验。

推荐使用支持断点续传和校验的受控方式，例如 rsync/对象存储/内部文件服务，
或 `release_artifacts.py upload/download`。Git 不承担数十 GB raw。

### 10.3 同步包

若团队另行建设对象存储，可为每个 capture 生成：

```text
<experiment-id>-reference-r1.compact.tar.zst
<experiment-id>-candidate-r1.compact.tar.zst
<experiment-id>-reference-r1.raw-index.json
<experiment-id>-candidate-r1.raw-index.json
```

这不是当前 workflow 自动生成的固定文件名。压缩包外另存 SHA-256，解包后重新
校验 manifest，防止只传了一部分目录。

## 11. force、继续运行和 generation

### 11.1 `--force`

`--force` 必须对用户选中的实验范围建立新一代：

1. 检查 active PID/进程组；
2. 拒绝删除仍在运行的目录；
3. 删除该 generation 的 fixture、run、artifact、report；
4. 创建新的 generation ID；
5. 从 data/capture 的必要阶段重跑；
6. 不允许旧 raw 与新 manifest 混搭。

如果只 `--force --capture candidate --topology tp8`，清理范围必须由 CLI 规则明确；不能意外删除其他拓扑已经完成的数据，也不能复用旧 candidate raw。

### 11.2 不加 `--force`

不加 force 是安全续跑：

- 完整且 hash/config/generation 匹配的阶段跳过；
- 失败或不完整阶段清理其临时/partial 输出后重跑；
- 旧 generation、不同 config 的结果拒绝复用；
- compare/report 可在 capture 不变时反复生成；
- `.status=running` 但 PID 已消失的 orphan run 标记 interrupted，再按规则重跑。

长时间 `acc_check` 的官方断点续检是显式例外：`--precheck-resume-csv` 只允许一个
topology/step/rank，要求成对 result/details CSV，并把两份源数据复制到当前 member
后续写。源路径与 SHA-256 进入 operation identity；默认续跑不会自动猜测任意旧
CSV。

capture manifest 会记录 lock、resolved manifest、实际安装树/CLI 与相关工具源码
身份，但这些字段是该次采集的不可变 provenance，不参与“当前环境下是否完成”的
判定。升级 msProbe、TorchTitan 或测试仓后，不加 `--force` 仍会跳过已经完整落盘的
旧 capture，绝不能因为当前环境变化而静默归档并重采。若需要用新工具重新建立证据，
必须显式 `--force` 建立新 generation，或使用新的实验 identity。msprof-analyze 属于
独立离线分析身份；只升级 analyzer 时保留 capture，只重跑明确选择的 analysis，不能
把旧 analyzer 输出混进新报告。

### 11.3 原子提交

当前 capture 直接写最终 run/artifact 目录，但只有子进程成功、关键官方文件校验、
manifest/hash 完成后才写 `complete.json`。中断目录不会被视为成功，下次运行会
先归档再重跑。`.partial-<attempt-id>` + 原子 rename 是后续可选增强，不应写成
当前已经采用的机制。

## 12. 存储控制

### 12.1 为什么全拓扑 × 全 preset 会产生数百 GB

性能或精度 raw 的体积近似：

```text
拓扑数 × preset 数 × 采集 rank 数 × active step 数 × 每 rank 事件/tensor 量
```

例如 27 个拓扑、8 个 preset、8 rank 的完整 profiling 不是“同一份 trace 换八种图”，而是大量独立 capture。distributed/system/memory/stack/shape/op-args 等 preset 可能改变采集层级和原始数据，数百 GB 是可能的，但通常说明实验矩阵过度采集，不应作为默认流程。

### 12.2 分层采集策略

- 全拓扑：profiler-off 基线或轻量 overview；
- 代表性拓扑：distributed/kernel/memory/runtime/system 深度 preset；
- 精度：先统计量/小 step/单 rank，再对异常模块 tensor dump；
- graph：先 compile log/tlparse，再对异常 graph 保存大 dump；
- operator：只对证据指向的 kernel 做 msOpProf。

### 12.3 配额和预估

每次启动前输出：

- 展开的 capture 数；
- 拓扑/preset/rank/step；
- 历史同 preset 的单 run 体积；
- 预计总量和剩余磁盘；
- raw retention 策略。

超过配额时默认拒绝，要求用户显式确认大规模采集，而不是静默填满挂载盘。

## 13. Git、Release 和长期保留

### 13.1 Git

代码仓库默认跟踪：

- 代码和文档；
- toolchain lock。

`mindstudio_fixtures/`、`mindstudio_runs/`、`mindstudio_artifacts/` 整体由
`.gitignore` 忽略，因此其中即使是小型 manifest/config/summary 也不会自动进入
Git。`mindstudio_reports/` 没有被整体忽略，可在审查敏感信息和文件体积后选择
提交；更大的轻量分析集合应通过 Release `analysis` 或受控 wiki assets 传输。

默认忽略：

- `mindstudio_runs/`；
- raw tensor/profile/checkpoint；
- compiler cache；
- 大型数据库和 TensorBoard event；
- 构建 wheel 和外部源码。

### 13.2 GitHub Release

当前 `release_artifacts.py` 已把 `mindstudio_fixtures`、`mindstudio_runs`、
`mindstudio_artifacts`、`mindstudio_reports` 纳入发现与恢复测试：

```bash
# 推荐：只读分析/分享用的轻量集合
python release_artifacts.py upload <experiment-id> --content analysis

# 只有无损续跑/重新解析时才使用
python release_artifacts.py upload <experiment-id> --content full

python release_artifacts.py download <experiment-id> --overwrite
```

`--content` 的 CLI 默认值仍是 `full`，这是兼容与无损恢复语义，不是日常推荐值。
`analysis` 保留报告、紧凑 manifest/metrics/log、CSV/JSON/XLSX/HTML/MD 和处理后的
`.vis.db`，排除 raw tensor、checkpoint、CANN 原始采集树和编译 cache；它不保证
包含重新运行 tensor compare 所需 raw。`.vis.db` 虽是处理后数据，仍可能包含模块
名、统计量、源码栈和绝对路径。full 还可能包含 tensor、token、checkpoint 和完整
runtime。两者上传前都要审查；大型 raw 优先使用内部对象存储或 rsync，不应因为
collector 支持就默认传到 GitHub。

推荐的轻量 release 内容：

- 中文 index/report；
- compact artifacts；
- toolchain/config/fixture identity；
- 小型官方 compare/analyze 输出；
- 经审查的 `graph_visualize` `.vis.db`；
- raw index 与受控存储说明。

大型 raw 只在内部对象存储/文件服务器保留，release 中提供 hash 和访问说明。

### 13.3 保留等级

| 等级 | 内容 | 建议保留 |
| --- | --- | --- |
| L0 | runtime 临时文件、cache、partial | 失败定位完成后清理 |
| L1 | raw tensor/profile | 受控短期保留，按容量策略 |
| L2 | official compare/analyze | 项目周期内长期保留 |
| L3 | compact artifact、report、lock | 长期保留并可发布 |

## 14. 一次实验的推荐阅读顺序

1. 打开 `mindstudio_reports/<experiment-id>/README.md` 看目的和运行状态；
2. 打开 `<experiment-id>.html` 看契约、端点、拓扑和工具版本；
3. 先读官方 msProbe compare 摘要，再看 loss/grad norm；
4. 若只在 compile 异常，查看 graph break/recompile/FX/IR/code；
5. 若精度正确但性能异常，查看 profiler summary、advisor、cluster；
6. 需要交互分析时，把 manifest 指定的 Insight 数据导入兼容桌面版；
7. 需要复核时，依据 manifest/hash 找 raw，而不是从截图反推；
8. 最后读“人工结论与下一步”，区分已证实根因和待验证假设。

## 15. 完整性验收清单

- [ ] fixture、reference、candidate 属于同一 config/generation；
- [ ] 每个阶段有命令、runtime log、exit code；
- [ ] 官方配置文件已复制并 hash；
- [ ] toolchain source/version 可追溯；
- [ ] official raw 有实际索引，不是空目录；
- [ ] compact artifact 解包后附件校验通过；
- [ ] compare 不依赖原服务器绝对路径；
- [ ] 中文报告的链接目标存在或明确标记 unavailable；
- [ ] 阶段完成状态与精度 PASS/FAIL 分开；
- [ ] raw 体积、保留位置和敏感性已说明；
- [ ] force/续跑未混用新旧 generation；
- [ ] `analysis` release 不意外携带 checkpoint、raw tensor/CANN collection 或凭据；
- [ ] `.vis.db` 已审查模型名、统计量和源码/服务器路径；
- [ ] `full` 仅在明确需要无损恢复时使用。
