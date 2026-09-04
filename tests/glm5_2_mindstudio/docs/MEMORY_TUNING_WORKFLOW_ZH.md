# GLM5 内存调优：msMemScope 与 MindStudio Insight

## 1. 与系统 Memory 页的边界

Ascend PyTorch Profiler 的 Memory 页适合把训练时间线上的内存曲线、算子申请和峰值联系起来；msMemScope 是独立内存事件工具，通过 hook 采集 alloc/free/launch/access、Python/C 调用栈和对象归属，并提供泄漏、拆解、低效内存、内存块监测和 step 对比。两套证据互补，但不能共用一次采集冒充另一种数据。

官方依据：

- [Insight 内存调优](https://www.hiascend.com/document/detail/zh/mindstudio/latest/GUI_baseddevelopmenttool/MindStudioInsight/docs/zh/user_guide/memory_tuning.md)
- [msMemScope 快速入门](https://www.hiascend.com/document/detail/zh/mindstudio/latest/msTT_msIT/msMemScope/docs/zh/quick_start/quick_start.md)
- [msMemScope 采集参数](https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/user_guide/memory_profile.md)
- [泄漏、对比、拆解和低效内存](https://gitcode.com/Ascend/msmemscope/blob/master/docs/zh/user_guide/memory_analysis.md)

## 2. 安装和环境

msMemScope 不等同于 CANN 自带的系统 profiler。应按目标架构和 CANN 版本执行官方仓库安装流程；源码 bootstrap 只取得源码，不猜测系统 hook 库安装位置。

```bash
python tools/bootstrap_mindstudio_toolchain.py \
  --root /home/$USER/mindstudio-tools --tool msmemscope

# 随后在源码目录按官方 install guide 安装并加载环境脚本。
source /usr/local/Ascend/ascend-toolkit/set_env.sh
which msmemscope
msmemscope --version
python -m tests.glm5_2_mindstudio.toolchain doctor --scope memory

sqlite3 --version
# Ubuntu 缺失时：sudo apt-get install sqlite3 libsqlite3-dev
```

若官方安装没有提供一键环境脚本，按实际安装根配置（文件必须真实存在）：

```bash
export MSMEMSCOPE_ROOT=/abs/path/to/installed/msmemscope
export LD_LIBRARY_PATH="${MSMEMSCOPE_ROOT}/lib64:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${MSMEMSCOPE_ROOT}/lib64/libascend_kernel_hook.so:${MSMEMSCOPE_ROOT}/lib64/libascend_mstx_hook.so:${MSMEMSCOPE_ROOT}/lib64/libatb_abi_0_hook.so:${MSMEMSCOPE_ROOT}/lib64/libatb_abi_1_hook.so:${MSMEMSCOPE_ROOT}/lib64/libleaks_ascend_hal_hook.so"
export PYTHONPATH="${MSMEMSCOPE_ROOT}/python:${PYTHONPATH:-}"
```

上述库必须来自同一安装版本，不要混用其他版本。需要 workspace 采集时，`TASK_QUEUE_ENABLE=2` 会改变可见信息；做版本/step 对比时必须保持一致。退出专项调优 shell 后应恢复原 `LD_PRELOAD`，避免影响普通训练。

## 3. GLM5 标准采集

### 3.1 单卡完整内存诊断

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
python tests/glm5_2_mindstudio/memory_tuning_benchmark.py \
  --probe --topology single \
  --steps 10 --local-batch-size 8 --global-batch-size 64 \
  --sequence-length 128 --seed 61
```

默认采集 `alloc,free,launch,traceback`，保留 Python/C 调用栈，执行泄漏与组件拆解，输出 Insight 可导入的 DB。训练拆解可识别 weight、gradient、optimizer_state 和 ATen activation 等归属。

### 3.2 分布式与 all

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tests/glm5_2_mindstudio/memory_tuning_benchmark.py \
  --probe --topology fsdp8
python tests/glm5_2_mindstudio/memory_tuning_benchmark.py \
  --probe --topology all
```

`all` 沿用 common 拓扑，但内存 hook 与调用栈开销显著。应先 single，再选择有明确问题的拓扑，不建议无问题地全拓扑深采集。

### 3.3 指定 step 或 kernel

```bash
python tests/glm5_2_mindstudio/memory_tuning_benchmark.py \
  --probe --topology single \
  --level kernel --capture-steps 3 \
  --events alloc,free,launch --analysis ''
```

官方限制：内存分析不能和 `--steps` 同开；两个 step 对比应分别采集单个 step，再对两份结果执行 compare。CLI 会直接拒绝不合法组合。

### 3.4 官方内存对比

```bash
python tests/glm5_2_mindstudio/memory_tuning_benchmark.py \
  --compare /abs/path/to/baseline/memory_profile \
            /abs/path/to/candidate/memory_profile \
  --level kernel
```

用于相同训练参数的不同 step、软件版本或优化前后对比；只有输入契约一致时才能归因。

## 4. 输出和导入

```text
mindstudio_runs/memory/<N-card>/<topology>/<experiment>/
├── runtime.log
├── training.log
├── run_state.json
├── manifest.json
├── trainer_output/
└── memory_profile/                 # 官方 msMemScope 输出根
    ├── *.db                        # 导入 Insight
    └── *.csv                       # 文本检查

mindstudio_runs/memory/compare/<experiment>/memory_compare/
└── compare/memory_compare_*.csv
```

终端会打印准确路径。Insight 导入 DB 文件/官方内存输出根后进入独立内存调优页面，而非系统调优 Memory 页。

## 5. 问题到功能的选择

| 现象 | 采集/分析 |
| --- | --- |
| step 后显存持续上涨 | `events=alloc,free,traceback` + `analysis=leaks` |
| 不清楚显存由谁占用 | `level=op` + `analysis=decompose` |
| 相同配置两个 step 不一致 | 分别采两个 step，再 `--compare` |
| 频繁小块申请释放 | 统计 alloc/free 次数、大小和调用栈 |
| 临时 Tensor 长期闲置 | `analysis=inefficient`，先核对支持范围 |
| 内存踩踏 | Python API/watch；通常需 `ASCEND_LAUNCH_BLOCKING=1` |

## 6. Insight 阅读顺序

1. 申请/释放曲线确认峰值、阶梯增长和 step 边界；
2. 内存块图找到大块、长寿命和未释放对象；
3. 调用栈火焰图定位 Python/C 分配来源；
4. 拆解图比较 weight、gradient、optimizer、activation/workspace；
5. 对比表确认新增、消失、大小或寿命改变的事件；
6. 回到代码检查对象引用、缓存、checkpoint/recompute 和 optimizer 生命周期。

最终验收同时比较 peak active/reserved、碎片、step time、吞吐和精度。

## 7. 参数摘要

| 参数 | 默认 | 作用 |
| --- | --- | --- |
| `--topology` | `single` | common 单拓扑或 `all` |
| `--level` | `op` | op 或 kernel 粒度 |
| `--events` | `alloc,free,launch,traceback` | 内存和调用链事件 |
| `--call-stack` | `python:20,c:20` | 调用栈类型与深度 |
| `--collect-mode` | `immediate` | 全程或配合 Python API 的 deferred |
| `--analysis` | `leaks,decompose` | 官方分析功能组合 |
| `--data-format` | `db` | DB 用于 Insight，CSV 用于文本检查 |
| `--capture-steps` | 空 | 最多五个官方 step ID |
| `--compare A B` | 空 | 对两份既有输出执行官方比较 |
| `--tool-arg=...` | 空 | 当前版本确认过的额外官方参数 |
| `--force` | 关闭 | 删除所选 generation 后完整重跑 |
