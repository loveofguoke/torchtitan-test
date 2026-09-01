# MindStudio 官方标准工作流实施计划

本文只描述 `tests/glm5_2_mindstudio`。阶段定义、工具选择、输入格式和结果解释均以
MindStudio 26.1 官方文档为准。可执行命令以 [README](../README.md) 和各 CLI 的
`--help` 为准；服务器未验证的阶段必须显示为未执行，不能推断为 PASS。

## 1. 目标工作流

```text
确定问题现象
  -> 固定源码、软件栈、训练参数、输入和随机性
  -> 配置 CheckList 与最小规模复现
  -> 选择精度、性能、编译、内存或算子入口
  -> 用官方工具采集原始证据
  -> 用官方 CLI/GUI 分析并定位首个异常层级
  -> 缩小 step/rank/module/API/operator 后重新采集
  -> 修改单一变量并执行相同契约的 A/B 验收
```

工作流必须保存：准确命令、runtime log、源码和工具版本、输入契约、官方原始产物、
官方分析产物、可视化导入根、人工结论和后续动作。

## 2. 当前已实现的训练工具链

### 2.1 精度与配置

- ConfigChecker：采集并比较环境变量、三方库、启动参数、权重和数据；
- 确定性契约：固定模型初始化、数据顺序和算子/通信随机性；
- PrecisionDebugger：L0/L1/mix、statistics/tensor、前向/反向和分布式逐 rank dump；
- `msprobe compare`：生成官方比较表和专家建议；
- `acc_check`、`multi_acc_check`、`api_precision_compare`：API 预检及两端预检结果比较；
- `graph_visualize`：从 `construct.json` 生成 `.vis.db`，供 TensorBoard Ascend Graph 使用；
- TrainerMonitorV2：长时采集激活、梯度、权重和优化器状态统计；
- PrecisionChecker：同一 NPU 进程内执行 eager/compile single-pass 前向和反向比较。

### 2.2 性能

- msProf：从进程外包装短训练，采集 CANN/Device/通信数据库；
- Ascend PyTorch Profiler：从训练 step 生命周期采集 Framework、CANN、Device、shape、
  stack、memory、module 和通信证据；
- msprof-analyze advisor：对 `*_ascend_pt` 执行专家规则诊断；
- msprof-analyze cluster：支持 `all`、`communication_time`、
  `communication_matrix`、`--agent` 和受控的分析器 `--force`；
- msprof-analyze compare：对官方支持的 profile 执行计算、通信、调度、算子和内存比较；
- MindStudio Insight handoff：索引 Summary、Timeline、Communication、Operator、Memory
  所需的完整导入根和便携路径。

### 2.3 生命周期与证据完整性

- 独立 `mindstudio_fixtures/runs/artifacts/reports`；
- fixture generation、capture attempt、工具版本和文件 hash；
- `--force` 在启动前清理完整选中 generation；
- 无 `--force` 只复用同一 generation 中已完成且校验通过的成员；
- active PID 保护和中断恢复；
- 每个子进程都打印并记录命令与日志路径；
- 原始采集、离线分析和 GUI handoff 使用不同身份，升级分析工具不要求重跑训练。

## 3. 官方问题分流

### 3.1 训练精度

1. 执行 CheckList：训练参数、环境变量、三方库、数据、模型结构和初始权重；
2. 固定随机性、关闭数据 shuffle，按需要关闭 dropout，并开启算子和通信确定性；
3. 先用 Monitor V2 观察异常 step/rank/layer；
4. 稳定复现后从 L0 statistics 开始；
5. 找到可疑模块后缩小到 L1/mix；
6. 最后只对可疑 API 保存 tensor 并执行 API 预检；
7. 用 graph_visualize 检查结构、首个高风险节点和通信收发关系。

### 3.2 编译精度

同一 NPU 软件栈、checkpoint、输入和拓扑下比较 eager 与 compile。先用
PrecisionChecker single-pass 检查被编译模块的前向和反向，再按官方结果定位模块；
不能用 CSV 存在、标题行或 SKIP 行代替有效数值记录。

### 3.3 性能

1. 用无采集短训练确认拓扑可运行并取得性能基线；
2. 用 overview 短窗口采集全 rank 的 Level1 证据；
3. 导入 Insight Summary，先把问题定界为计算、调度或通信；
4. 计算问题进入 Operator 并按算子类型、名称和输入 shape 比较；
5. 调度问题进入 Timeline，检查 HostToDevice、Free 和下发间隙；
6. 通信问题进入 Communication，检查通信域、同步/等待时间和通信矩阵；
7. 内存异常进入 Memory，检查 active/reserved、分配释放、碎片和内存重整；
8. advisor 只给候选建议；cluster/compare/Insight 原始证据用于验证；
9. 每次只修改一个因素并重复相同采集窗口。

## 4. 当前适用边界

当前自动化面向 TorchTitan GLM-5.2 训练。MindStudio 目录中的推理和算子文档用于选择
正确的后续工具，但下列流程尚未伪装为已自动化：

- 推理模型转换、量化、服务化部署和 AISBench；
- 自定义算子的工程生成、仿真、Sanitizer、单算子 msOpProf 和生产交付；
- msMemScope 对任意进程的注入式全自动编排；
- msServiceProfiler 的推理服务启动和请求回放。

这些能力必须在确定目标模型格式、服务进程、算子工程和工具版本后增加专用 adapter。
在此之前只提供官方入口、安装说明和产物交接，不生成未经验证的命令。

## 5. 验收条件

一个阶段只有同时满足以下条件才算“工具阶段完成”：

1. doctor 证明目标 CLI/import 与 CANN/torch_npu 版本匹配；
2. 命令、环境和日志被记录；
3. 输入目录满足官方格式和同一次采集要求；
4. 官方 CLI 返回成功；
5. 文档要求的关键产物真实存在且可解析；
6. artifact 保存文件 hash、工具版本和 capture identity；
7. Insight handoff 指向完整导入根；
8. 结论明确区分“工具执行成功”“指标通过”“已定位根因”。

数值或性能结论仍需阅读官方表格/数据库和可视化页面；工具执行成功不能自动生成
模型正确或性能达标的结论。

## 6. 官方依据

- [MindStudio 使用导读](https://www.hiascend.com/document/detail/zh/mindstudio/latest/index/index.html)
- [大模型训练精度定位指南](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/LargeModelTrainingAccuracy/docs/zh/best_practices/train_debug_guide.md)
- [大模型训练性能瓶颈定位案例](https://www.hiascend.com/document/detail/zh/mindstudio/2610/practicalcases/Largemodeltraining/MindStudio/26.1.0/zh/cases/case_of_troubleshooting_performance_bottleneck_in_llm_training.md)
- [msprof-analyze 集群分析](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msprof_analyze/docs/zh/user_guide/cluster_analyse_instruct.md)
- [MindStudio Insight 概述](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/overview.md)
- [官方文档与能力矩阵](OFFICIAL_DOCUMENTATION_MATRIX_ZH.md)
