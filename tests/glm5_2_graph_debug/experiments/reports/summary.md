# GLM-5.2 图模式调试总结

## 最终状态

| 后端 | 正式 smoke | 结论边界 |
|---|---:|---|
| Inductor | 15/15 PASSED | 原生 Inductor/AOT/Triton 编译执行，10/10 steps |
| NPUGraphs profile | 15/15 PASSED | AOT 编译执行；`NPUGraph replay` 全局禁用 |
| NPUGraphs native replay | 未通过 | grouped-mm capture 和 TP `DeviceMesh` 仍不兼容 |

软件组合为 CANN 9.1.0、torch `2.14.0.dev20260805+cpu`、torch_npu `2.14.0`、
triton_ascend `3.2.1`。PyTorch 没有升级或降级。

## 用户列出的十项是否实现

全部已实现，不是待办：

| 项目 | 默认实现 | 代码位置 |
|---|---|---|
| 隔离 CANN 9.0 污染 | `env -i` 后在 Python 前 source CANN 9.1 | `graph_env_common.sh` |
| 独立 Conda/ATB/HCCL | clone Conda、ATB ABI 路径、动态 host 端口、600/2400 秒超时 | `graph_env_common.sh` |
| 仓库外编译缓存 | Inductor/Triton/compile debug 指向主目录 `.cache` | `graph_env_common.sh` |
| Inductor fallback | `aten.sum,_c10d_functional.all_reduce` | `graph_env_common.sh` |
| TP `aten.complex` strategy | launcher-only DTensor pointwise strategy 注册 | `train_npu.py` |
| PP metadata 通信 | metadata object P2P 改为 `use_batch=False` | `train_npu.py` |
| EP 空 expert/grouped-mm | 空组补行/全空 bypass/显式 backward/int32 offsets | `train_npu.py` |
| 零长度 Triton kernel | base 与 symbolic grouped autotuner 的 `*numel=0` guard | `train_npu.py` |
| NPUGraph replay 默认关闭 | `TORCHTITAN_NPUGRAPH_SKIP_ALL=1` | common + `train_npu.py` |
| NPUGraph task queue 0 | 降级 profile 默认 `TASK_QUEUE_ENABLE=0` | `graph_env_common.sh` |

这些 patch 都由 `GRAPH_*` 环境变量控制，只在 graph-debug launcher 子进程中启用。
普通训练不设置这些变量时不会自动开启对应兼容项。

## 为什么 CANN 9.1.0 仍不够

CANN 9.1.0 解决了 torch_npu 2.14 所需 ACL header 类型缺失，是必要条件。但后续
TP/PP/EP、collective 和 NPUGraph capture 问题发生在 PyTorch、torch_npu、Triton、
HCCL 与模型动态 shape 的组合边界，因此仍需要上表处理。详细演进见
[failures.md](failures.md)。

## 产物边界

- Git 内：`tests/glm5_2_graph_debug/` 的脚本、安装说明、实验索引和报告。
- Git 忽略：`graph_debug_runs/`、`smoke_runs/` 的运行日志和 trainer 输出。
- 仓库外：`~/.cache/torchtitan-test/graph_mode/` 的编译中间文件。
- 未执行 `git add`、`git commit` 或 `git push`。
