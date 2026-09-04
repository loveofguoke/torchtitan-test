# GLM5 算子调优：msOpProf 与 MindStudio Insight

## 1. 与系统调优的边界

系统调优先用 Ascend PyTorch Profiler/msProf 找到整步关键路径和热点算子；算子调优再把一个已确定的热点缩成可重复的单算子或少量算子程序，用 `msopprof` 采集核内性能。两者数据格式、采集开销和 Insight 页面不同。

```text
整网系统调优：step -> ATen/aclnn -> device op/task -> 通信与空闲
单算子调优：   一个/少量 kernel -> Cube/Vector/MTE -> Cache/UB/GM -> 源码/指令
```

官方依据：

- [Insight 算子调优快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/quick_start/operator_tuning_quick_start.md)
- [Insight 算子调优界面](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/operator_tuning.md)
- [msOpProf 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msOT/Operatordevelopmenttools/docs/zh/quick_start/msopprof_quick_start.md)
- [msOpProf 源码和完整参数](https://gitcode.com/Ascend/msopprof)

## 2. 环境

`msopprof` 通常随匹配 CANN 的算子开发工具提供。源码 checkout 用于阅读实现，不代表替换服务器中与 CANN 配套的二进制。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which msopprof
msopprof --version
python -m tests.glm5_2_mindstudio.toolchain doctor --scope operator

python tools/bootstrap_mindstudio_toolchain.py \
  --root /home/$USER/mindstudio-tools --tool msopprof
```

源码热点映射需给 Kernel 编译添加 `-g`，性能测量仍应保留优化编译，不能拿 `-O0` 结果代表生产性能。具体指标受芯片与版本约束；不支持的指标应由当前 `msopprof --help` 明确拒绝，框架不会偷偷降级。

## 3. 标准流程

先从系统数据记录算子名、输入 shape/dtype、调用次数、总耗时和训练阶段，再编写只加载真实实现和典型 shape 的小程序。它可以是 Ascend C/aclnn 可执行文件，也可以是稳定触发目标 Kernel 的 PyTorch/Triton 脚本。

### 3.1 上板采集

```bash
python tests/glm5_2_mindstudio/operator_tuning_benchmark.py \
  --probe --name glm5-dsa-topk --mode onboard \
  --kernel-name 'TopK*' --launch-count 5 --metrics Default \
  -- python /abs/path/to/glm5_dsa_topk_probe.py
```

`Default` 一次取得 Arithmetic、L2、Memory、MemoryL0、MemoryUB、PipeUtilization 和资源冲突等官方基础指标。针对 A2/A3 且目标 API 满足官方限制时可单独深采集：

```bash
python tests/glm5_2_mindstudio/operator_tuning_benchmark.py \
  --probe --name glm5-dsa-topk-detail \
  --kernel-name 'TopK*' --launch-count 1 \
  --metrics TimelineDetail,Default --dump \
  -- python /abs/path/to/glm5_dsa_topk_probe.py
```

Roofline、Occupancy、MemoryDetail、Source 等按问题选择，避免机械组合互斥或芯片不支持的选项。Triton 算子不支持的 TimelineDetail 路径不能伪装成功。

### 3.2 仿真采集

仿真用于指令流水和源码热点，不替代真实上板带宽与耗时：

```bash
python tests/glm5_2_mindstudio/operator_tuning_benchmark.py \
  --probe --name glm5-dsa-topk-sim \
  --mode simulator --soc-version Ascend910B2 \
  --kernel-name 'TopK*' --launch-count 1 --metrics Default \
  -- python /abs/path/to/glm5_dsa_topk_probe.py
```

也可以传 `--config-file operator.json` 直接分析算子 `.o` 配置。

## 4. 输出和导入

```text
mindstudio_runs/operator/<npu-name-mode-hash>/
├── runtime.log
├── run_state.json
├── manifest.json
└── operator_profile/
    └── OPPROF_<timestamp>_<id>/    # 官方目录，原样保留
        ├── *.csv
        ├── dump/
        ├── visualize_data.bin
        └── trace.json              # 取决于模式与指标
```

终端会打印 `MindStudio Insight operator import:`。导入其中的 `visualize_data.bin` 或官方指定的 OPPROF 数据；不要导入系统 profiler 的 traces 目录来期待算子调优页面。

## 5. 阅读与验收

1. OpBasicInfo 确认 shape、dtype、block 和核型；
2. Pipe/Arithmetic 判断 Cube、Vector、Scalar、MTE 主导项；
3. Memory/MemoryL0/MemoryUB 检查 GM、L2、L1、L0、UB 搬运；
4. L2Cache 与 ResourceConflict 检查回源、同步和资源冲突；
5. Roofline/Occupancy 判断上限与核间均衡；
6. Timeline/Source 把长气泡和热点映射到指令与源码。

优化后必须用同 shape、dtype、warmup 和设置重测，并回到无 profiler 整网训练验证 step time、精度、显存和稳定性。

## 6. 参数摘要

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--mode` | `onboard` | 上板真实测量或 simulator |
| `--metrics` | `Default` | 官方 AIC 指标组合 |
| `--kernel-name` | 空 | Kernel 名过滤 |
| `--launch-count` | `1` | 最多采集多少次匹配 launch |
| `--launch-skip` | `0` | 跳过前若干 launch |
| `--kill` | 关闭 | 达到采集数量后停止应用 |
| `--core-id` | 空 | 深度分析时限制逻辑核 |
| `--dump` | 关闭 | TimelineDetail 配套 dump，受芯片限制 |
| `--tool-arg=...` | 空 | 传递当前版本已确认的官方参数 |
| `--force` | 关闭 | 清理同一 generation 后重跑 |
