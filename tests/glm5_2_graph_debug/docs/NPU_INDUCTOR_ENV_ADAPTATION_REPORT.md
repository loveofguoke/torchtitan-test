# NPU 单卡 Inductor 图模式软件环境适配报告

## 摘要

`glm5-npu-dev` 中的 TorchTitan GLM-5.2 单卡 Inductor 图模式已经跑通。适配过程
保持 PyTorch、torch_npu、Triton-Ascend、Python 和 Driver 版本不变，仅为测试
进程将用户态 CANN 从 9.0.0 切换到独立安装的 CANN 9.1.0。

本文记录的是图模式软件栈调试与功能性验证，不替代仓库各正式实验的验收报告。

最终验证组合：

| 组件 | 版本/路径 |
|---|---|
| Conda | `/root/miniconda3/envs/torchtitan-0803-graph-adapt` |
| Python | 3.12.13 |
| torch | `2.14.0.dev20260805+cpu` |
| torch_npu | `2.14.0` |
| triton_ascend distribution | `3.2.1` |
| triton distribution metadata | `3.5.0` |
| `triton.__version__` | `3.2.0` |
| CANN | `/usr/local/Ascend/cann-9.1.0` |
| Driver | `26.0.rc1` |
| NPU | Ascend 910B2 |

## 根因

`torch_npu 2.14.0` 的 `AclInterface.h` 使用了：

```text
aclmdlRICondHandle
aclmdlRICondTaskParams
```

CANN 9.0.0 的 `acl_rt.h` 没有定义这两个类型。Triton-Ascend 在生成 launcher
预编译头时调用 g++，编译因此失败，最终被 Inductor 包装成：

```text
torch._inductor.exc.InductorError
RuntimeError: No valid triton configs
```

所以第一个真实 blocker 是 `torch_npu 2.14.0` 与 CANN 9.0 ACL header API
不匹配，而不是 PyTorch 2.14 本身。

CANN 9.1.0 的 `acl_rt.h` 已提供上述类型和 API。使用失败现场的同一份
`precompiled.h`，仅将 include/link 从 CANN 9.0 改为 9.1 后即可成功编译。

## 隔离方案

1. 使用 `conda create --clone` 从 `torchtitan-0803` 创建
   `torchtitan-0803-graph-adapt`。
2. 从同宿主已有的 `yhr_cann9.1.0` 容器复制 CANN 9.1.0 GA 到
   `/usr/local/Ascend/cann-9.1.0`。
3. 不修改 `/usr/local/Ascend/cann` 和 `ascend-toolkit/latest`，二者仍指向 9.0。
4. 每次测试通过 `env -i` 重建最小环境，避免 9.0 的 include/lib 混入。
5. 为 Inductor、Triton、编译诊断和运行日志使用独立目录。

完整安装与恢复步骤见 `CANN_9_1_INSTALLATION.md`。

CANN 复制后关键文件与来源 SHA-256 完全一致：

| 文件 | SHA-256 |
|---|---|
| `compiler/version.info` | `f20359296d5a7127f841aa35f4a788137f444dc36d72641e948cb9e39e47e4f4` |
| `include/acl/acl_rt.h` | `22c603cb80ab582006d4f26697ceb14c059d7f3613cf13b3c72aeb0d3d36e3c1` |
| `lib64/libascendcl.so` | `123d67c313f743e5d6f2e856f40c24d5edac376975cf0a7a1019e079b4171480` |
| `set_env.sh` | `de50f1bf5b960f8305792ef407e3bfc3fc1b6d82931d0feae39ac4c965690aaf` |

## 测试矩阵

| 编号 | 测试 | 结果 | 关键结果 |
|---|---|---:|---|
| 01 | CANN 9.0 冷缓存基线 | FAIL（预期） | 复现两个 ACL 类型未声明 |
| 02 | CANN 9.1 Eager 冒烟 | PASS | NPU 计算结果 140.0；无 9.0 动态库映射 |
| 03 | 原失败 launcher PCH + CANN 9.1 | PASS | g++ 退出码 0；PCH 497,444,247 bytes |
| 04 | CANN 9.1 冷缓存完整图模式 | PASS | 10/10 step；forward/backward/optimizer |
| 05 | CANN 9.1 同缓存复跑 | PASS | 10/10 step；80 秒；平均约 173.2 tokens/s |
| 06 | 新 `run_npu_inductor.sh smoke` 入口 | PASS | 10/10 step；61 秒；自动生成日志与报告 |
| 07 | 缓存/结果迁出仓库后的默认入口 | PASS | 10/10 step；84 秒；编译后端错误 0 |
| 08 | 最终 `glm5_2_graph_debug` 目录规范 | PASS | 10/10 step；59 秒；日志/报告/缓存路径全部符合约定 |

冷缓存完整测试约 7 分 35 秒，产生 5,546 个 `.npubin`（含 autotune 候选），
独立编译/诊断目录约 977 MB。冷缓存和同缓存复跑均未出现 graph break、
`BackendCompilerFailed`、`InductorError` 或 `No valid triton configs`。

## 测试命令

完整图模式验证采用现有 `compile_probe.py` 的 10-step 快速规格：

```bash
NGPU=1 LOG_RANK=0 MODULE=glm5 CONFIG=glm5_debugmodel \
TORCHTITAN_DEVICE=npu ASCEND_RT_VISIBLE_DEVICES=2 \
./run_train.sh \
  --training.steps=10 \
  --training.num_tokens_per_microbatch_per_dp_rank=32 \
  --training.num_tokens_per_train_step=32 \
  --training.max_context_length=32 \
  --training.dtype=float32 \
  --training.mixed_precision_param=bfloat16 \
  --compile.enable \
  --compile.components=model \
  --compile.backend=inductor
```

推荐使用本目录的新入口，它会自动完成环境隔离、缓存和报告生成：

```bash
tests/glm5_2_graph_debug/run_npu_inductor.sh smoke
```

## 环境未受影响的证据

- 原环境 `/root/miniconda3/envs/torchtitan-0803` 未执行安装或卸载。
- 原环境和克隆环境最终 `pip freeze --all` SHA-256 均为
  `0a9745072e856af14d4c567e05c32631b2d0433e2ee1cc12f2b2c8cba45b8463`。
- `/usr/local/Ascend/cann` 仍解析到 `/usr/local/Ascend/cann-9.0.0`。
- `/usr/local/Ascend/ascend-toolkit/latest` 仍解析到 CANN 9.0.0。
- Driver 仍为 `26.0.rc1`。
- 成功训练日志中的 CANN 9.0 路径出现次数为 0。

## Triton-Ascend 3.2.2 结论

本次没有升级。当前 3.2.1 已完成 launcher、kernel、autotune、forward、backward
和 optimizer 全链路验证；同时 3.2.2 官方兼容矩阵没有列出本项目的定制
`torch_npu 2.14.0`。继续改变已工作的组件会扩大变量。如后续出现独立的
Triton codegen/autotune 问题，应在新的 Conda 克隆中单独评估 3.2.2。

## 本地证据

详细逐项报告和完整日志作为调试运行结果保存在：

```text
graph_debug_runs/environment-adaptation-20260824/
```

其中 `logs/01_baseline_cann90/runtime.log` 是原始失败日志，
`logs/04_cann91_triton321/runtime.log` 是冷缓存成功日志，
`logs/05_cann91_warm_cache/runtime.log` 是同缓存复跑成功日志。

新脚本自身的验证产物为：

```text
graph_debug_runs/smoke-20260824-110323-890483/logs/runtime.log
graph_debug_runs/smoke-20260824-110323-890483/reports/report.md
```

可复用的已验证编译缓存位于：

```text
/home/y50064852_yyb/.cache/torchtitan-test/graph_mode/cann91-torch214-triton321/
```

迁移后默认路径验证的自动报告位于：

```text
graph_debug_runs/smoke-20260824-110655-896478/reports/report.md
```

最终目录规范验证结果为：

```text
graph_debug_runs/smoke-20260824-112720-917238/logs/runtime.log
graph_debug_runs/smoke-20260824-112720-917238/reports/report.md
```
