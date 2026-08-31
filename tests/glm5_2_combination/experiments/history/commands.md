# 组合图模式实验命令账本

以下命令均在容器 `glm5-npu-dev` 的
`/workspace/y50064852_yyb/torchtitan-test` 执行。公共 launcher 会建立干净环境、加载
CANN 9.1.0、选择独立 Conda/ATB/HCCL 配置，并把编译缓存定向到仓库外 `.cache`。

## 1. 三仓与环境验证

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor env
```

本轮报告：

```text
graph_debug_runs/three-repo-validation-20260826/
  launcher-env-inductor-20260826-110714-4119609/reports/report.md
```

## 2. smoke 全拓扑

为避免旧 manifest 静默跳过，并保存编译诊断，本轮使用：

```bash
tests/glm5_2_graph_debug/run_graph_mode.sh inductor smoke \
  --topology all --compiler-diagnostics

tests/glm5_2_graph_debug/run_graph_mode.sh npugraphs smoke \
  --topology all --compiler-diagnostics
```

本轮正式报告：

```text
graph_debug_runs/three-repo-validation-20260826/
  smoke-suite-inductor-20260826-111320-4162002/reports/report.md
  smoke-suite-npugraphs-20260826-114532-1009046/reports/report.md
```

调试期间的两次失败调用也被保留：

```text
graph_debug_runs/three-repo-validation-20260826/
  smoke-suite-inductor-20260826-110730-4119670/reports/report.md
  smoke-suite-inductor-20260826-110855-4120250/reports/report.md
```

## 3. maintained 5000-step eager/Inductor 精度

推荐用可断点续跑的入口：

```bash
tests/glm5_2_combination/run_graph_precision_5000.sh inductor all
```

它严格等价于下面四个阶段，未传 `--steps`，因此使用 maintained 5000 steps；未设置
`REPEAT`，因此使用配置中的两个 repeat：

```bash
COMMON_ARGS="--topology all --objectives precision --reference-graph eager --candidate-graph inductor --profiler-preset off"

tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --data --data-device npu ${COMMON_ARGS}
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture reference ${COMMON_ARGS}
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --capture candidate ${COMMON_ARGS}
tests/glm5_2_graph_debug/run_graph_mode.sh inductor combination \
  --compare --require-all ${COMMON_ARGS}
```

如任务中断，可只恢复某个阶段；有效 artifact 会被自动跳过：

```bash
tests/glm5_2_combination/run_graph_precision_5000.sh inductor data
tests/glm5_2_combination/run_graph_precision_5000.sh inductor reference
tests/glm5_2_combination/run_graph_precision_5000.sh inductor candidate
tests/glm5_2_combination/run_graph_precision_5000.sh inductor compare
```

调试单 repeat 或单 topology 时显式设置环境变量，不改变 maintained 默认值：

```bash
REPEAT=1 TOPOLOGY=tp8 \
  tests/glm5_2_combination/run_graph_precision_5000.sh inductor candidate
```

正式精度默认不开 compiler diagnostics，避免数十万 step 产生冗余诊断；smoke 已保存完整
诊断。若定位编译问题，可对定向 topology 设置 `COMPILER_DIAGNOSTICS=1`。

本轮正式任务曾使用独立 tmux 会话持久运行，并在容器内等待已有 performance 任务
PID 462169 正常释放 8 卡；没有 kill 或抢占该任务：

```bash
tmux new-session -d -s graph_precision_5000_20260826 \
  "docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -e GRAPH_RUN_ROOT=/workspace/y50064852_yyb/torchtitan-test/graph_debug_runs/precision-5000-20260826 \
    glm5-npu-dev /bin/bash -c \
    'while kill -0 462169 2>/dev/null; do sleep 30; done; \
     cd /workspace/y50064852_yyb/torchtitan-test; \
     tests/glm5_2_combination/run_graph_precision_5000.sh inductor all'"
```

该会话现已结束。历史上用于观察的命令是：

```bash
tmux capture-pane -pt graph_precision_5000_20260826:0 -S -200
```

实际结果为 data PASS、eager reference r1/r2 PASS、candidate single-r1 在 deterministic
pointwise autotune 失败。完成 Turbo/common 的 G020 兼容验证后，从 candidate 阶段续跑：

```bash
tests/glm5_2_combination/run_graph_precision_5000.sh inductor candidate
tests/glm5_2_combination/run_graph_precision_5000.sh inductor compare
```

### 2026-08-31 恢复记录

恢复前先确认已有 eager r1/r2 均为完整 5000-step artifact，并检查 8 卡 owner：

```bash
ps -eo pid,ppid,stat,etime,pcpu,pmem,args
npu-smi info
```

当时 PID 2459052 正在运行另一套正式 self-consistency 5000-step 矩阵，不是残余进程，
因此没有 kill。其后已有合法 `all_preset_0831` 队列取得资源顺序，并在 profiler matrix
结束后调用同一套 precision candidate/compare。为防该队列中断导致精度任务遗失，本任务
建立独立 tmux server/socket 的断点接力；它等待上游 pane PID 退出后再执行：

```bash
tmux -L graphprecision0831 new-session -d \
  -s graph_precision_5000_0831 \
  /workspace/y50064852_yyb/.cache/torchtitan-test/graph_mode/\
precision-5000-20260831/queue_after_all_preset.sh
```

队列脚本属于仓库外 `.cache` 中间文件，不进入 Git。它不删除 artifact，不重新生成 eager
reference；candidate 会按完整 experiment identity 自动复用上游已完成项。过程日志：

```text
graph_debug_runs/precision-5000-20260831/queue.log
```

观察命令：

```bash
tmux -L graphprecision0831 list-sessions
tail -n 100 graph_debug_runs/precision-5000-20260831/queue.log
```

## 4. NPUGraphs 精度边界

```bash
tests/glm5_2_combination/run_graph_precision_5000.sh npugraphs all
```

当前 common 默认 `GRAPH_NPUGRAPH_SKIP_ALL=1`，上述命令只能验证 eager 与
NPUGraphs AOT-compatible 路径的精度，不能证明 native replay 精度。原生 replay 在
底层问题关闭前不进入正式 acceptance。

## 5. 静态与单元验证

```bash
bash -n tests/glm5_2_combination/run_graph_precision_5000.sh
pytest -q tests/unit_tests/test_glm5_2_graph.py \
  tests/unit_tests/test_glm5_2_smoke.py
git diff --check
```

本轮结果：test 仓单测 `21 passed in 0.13s`；Turbo 的
`tests/unit_tests/test_graph_compat.py` 为 `4 passed in 8.43s`；shell 语法与两个工作区的
`git diff --check` 均通过。容器内运行 pytest 时必须显式使用对应仓库工作目录（例如
`docker exec -w /workspace/y50064852_yyb/torchtitan-test ...`），否则 pytest 只会报告找不到
相对测试路径，并未真正执行测试。

本任务不执行 `git add`、`git commit` 或 `git push`。

## 6. eager/Inductor 组合性能矩阵

性能任务的固定输入、eager/Inductor 全拓扑 batch、定向探针、正式 launcher、离线报告
生成和单拓扑复跑命令较长，单独完整保存在
[performance-commands.md](performance-commands.md)。本账本不复制它们，以免后续修改时
两个副本不一致。
