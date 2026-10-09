# 图模式失败与修复历史

本文件保留失败，因为失败是软件栈适配结论的一部分。最终通过不能抹掉曾经遇到的
编译器、通信、动态 shape、数值和资源问题。

本文件按时间保留现象、尝试和实验现场；面向 PyTorch、torch_npu、op-plugin、CANN、
HCCL 提单或制作 patch 时，请使用
[底层问题定位与提单交接](../../../glm5_2_graph/docs/LOWER_LAYER_ISSUE_HANDOFF.md)。后者进一步
记录具体文件、模块/函数、置信度、分界实验和 workaround-off 验收条件。

## G001：torch_npu 2.14 与 CANN 9.0 header 不匹配

- 现象：Triton launcher PCH 编译时 `aclmdlRICondHandle`、
  `aclmdlRICondTaskParams` 未定义，外层表现为 `InductorError`/无有效配置。
- 根因：CANN 9.0 `acl_rt.h` 不包含 torch_npu 2.14 使用的新 ACL 类型。
- 处理：独立安装 CANN 9.1.0；不修改全局 CANN 9.0 软链；从
  `torchtitan-0803` clone `torchtitan-0803-graph-adapt`。
- 复测：原失败 PCH 仅切换 9.1 include/lib 后编译通过；单卡 10/10 steps。
- 证据：`graph_debug_runs/environment-adaptation-20260824/`。

## G002：默认 shell 泄漏 CANN 9.0

- 现象：即使设置部分 9.1 变量，Python 导入期间仍可能解析 9.0 include/lib。
- 根因：`torch_npu` 在业务脚本解析前已经导入，脚本内晚切换无效。
- 处理：`graph_env_common.sh` 使用 `env -i` 重启干净 Bash，在 Python 前 source
  `/usr/local/Ascend/cann-9.1.0/set_env.sh`，再加入 Conda 和 ATB。
- 复测：最终 env report 的 CANN 标记全部为 9.1；运行日志无 9.0 动态库路径。

## G003：单卡 reduction 没有可用 Triton tile

- 现象：`triton_per_fused_add_mul_sum_29` 报 `No valid triton configs`；所有候选
  UB 需求 2,162,688 bit，高于设备 1,572,864 bit。
- 尝试：`triton_experimental` 绕过此点，但 softmax autotune 出现 507035 vector
  core exception，未采用。
- 处理：保留正式 `default` codegen，仅 fallback `aten.sum` 到 ACLNN。
- 失败：`smoke-suite-inductor-20260824-174131-1530285`、`...174646-1646481`。
- 通过：`smoke-suite-inductor-20260824-174917-1691961`，10/10 steps。

## G004：HCCL NPU NIC 默认端口 16666 冲突

- 现象：DDP8 `Communication_Error_Bind_IP_Port(EI0020)`。
- 根因：NPU NIC 属于宿主共享资源，其他容器可占用相同默认端口。
- 处理：`HCCL_NPU_SOCKET_PORT_RANGE=auto`，支持管理员覆盖固定范围。
- 失败：`smoke-ddp-20260824-160139-1440860`。
- 通过：`smoke-ddp-20260824-160345-1442387` 和后续正式 DDP8/FSDP8。

## G005：HCCL host socket 60000 冲突

- 现象：`EI0019`，`172.18.0.1:60000` 已绑定。
- 根因：host socket 与 NPU NIC socket 是两个独立端口面，仅设置 G004 不够。
- 处理：每次 wrapper 调用生成 62000–63584 范围内的 `HCCL_IF_BASE_PORT`；允许
  `GRAPH_HCCL_IF_BASE_PORT` 覆盖。
- 现场之一：`smoke-suite-npugraphs-20260824-175742-1799152`。

## G006：TorchTitanTurbo 把 task queue 重写为 2

- 现象：NPUGraph capture 明确拒绝 `TASK_QUEUE_ENABLE=2`。
- 根因：launcher 初始 export 会被 Turbo 通用 NPU patch 覆盖。
- 处理：common 同时传递 `TORCHTITAN_TASK_QUEUE_ENABLE`，`train_npu.py` 在导入
  Turbo 后恢复 profile 请求值。
- 失败：`smoke-suite-npugraphs-20260824-175231-1791953`、
  `...175336-1794734`。
- 最终策略：两个 graph profile 都使用 queue 0；原生 capture 调试显式用 1。

## G007：NPUGraph grouped-mm 捕获期间同步 device tensor

- 现象：`thread_local` 下 host/device copy 返回 107030；改为 `relaxed` 后 stream
  synchronize 返回 107027。
- 根因：`aten._grouped_mm` 的 offsets 路径在 capture 中执行同步操作；放宽 capture
  mode 不能消除同步。
- 历史第一阶段：扫描 FX 子图，对 grouped-mm 子图跳过 replay。
- 最终处理：TP 后续还暴露非算子级输入问题，因此设置
  `TORCHTITAN_NPUGRAPH_SKIP_ALL=1`，保留 AOT、关闭整个模型 replay。
- 失败：`...npugraphs-20260824-175610-1796475`、`...175742-1799152`、
  `...175916-1800868`、`...180931-1810661`、`...181321-1817359`。
- 第一阶段单卡通过：`...181603-1824452`；最终 profile 结论不是原生 replay。

## G008：TP 缺少 `aten.complex` DTensor strategy

- 现象：TP8 报 `Operator aten.complex.default does not have a sharding strategy`。
- 根因：NPU RoPE 对 DTensor 的实部/虚部调用 `aten.complex`，当前 torch 2.14 未注册
  对应 pointwise strategy。
- 处理：launcher-only 调用 PyTorch 的 broadcast-aware pointwise strategy 注册器。
- 失败：`smoke-suite-inductor-20260824-181711-1826722`。
- 通过：`smoke-suite-inductor-20260824-182622-1906926`。

## G009：PP metadata batched P2P 损坏 object size

- 现象：PP8 出现 `Tried to allocate more than 1EB`，随后 `EOFError`；不是正常 HBM
  不足。
- 根因：PyTorch `PipelineStage` 初始化时用 batched object P2P 交换 shape/dtype，
  当前 HCCL 路径接收到损坏的 serialized size。
- 处理：只对一次性 metadata 的 `send_object_list`/`recv_object_list` 使用
  `use_batch=False`，activation/gradient 通信不变。
- 失败：`smoke-suite-inductor-20260824-183751-3070299`。

## G010：PP 冷编译超过 HCCL 建链/进程组等待

- 现象：PP8 图模式首步中，下游 rank 在 `PipelineStage` metadata receive 等待约
  300 秒后超时；延长 timeout 只能推迟失败，不能消除首步串行等待。
- 根因：PyTorch 动态 metadata inference 会执行真实 stage forward。包含
  `torch.compile` 子模块时，rank 0 编译完成后才发送 metadata，rank 1 才开始编译，
  依次传播到 rank 7，八级冷编译被错误地串行放进 P2P 链路。2026-10-08 的全 rank
  日志中，rank 0-3 依次约在 15:38:01、15:38:09、15:38:18、15:38:26 产出编译
  结果，rank 4 到 15:42:18 才完成，直接解释了下游 300 秒 watchdog。
- 排除项：真实 `_StageForwardMeta` 的 8-rank 最小复现对
  `TASK_QUEUE_ENABLE={0,2}` × `use_batch={false,true}` 四种组合均通过，因此当前
  失败不是 metadata 对象本身无法传输；G009 的历史 1EB/EOF 与本问题分开记录。
- 官方语义边界：PyTorch `PipelineStage` 只允许两类 metadata 合同。静态模式由
  调用方通过 `input_args`、`output_args`（DTensor/反向场景还包括 grad metadata）
  提供完整合同；动态模式由真实上游 stage 运行 forward 并逐级传输 metadata。
  本实验继续使用动态模式，不能自行制造本地 metadata，也不能用 host store 替换
  原通信协议。
- 当前修复候选：仅在 PyTorch 原始
  `_forward_metadata_inference`/`_backward_metadata_inference` 执行期间进入官方
  `torch.compiler.set_stance("force_eager")` 上下文。真实 stage、真实输入、metadata
  内容、发送接收顺序和校验全部不变；只避免把冷编译串行塞进启动期 metadata P2P。
  当前安装的 PyTorch 尚无公开 `PipelineStage.register_forward_context`，所以 Turbo
  只提供一个可移除的兼容桥；升级到包含公开 hook 的版本后应迁移到公开 API。
- 已否决实验：本地 metadata 推断绕过了真实同步语义；host store 只改变等待位置；
  dummy-input 并行预编译需要复制输入/反向/状态恢复合同。这些都不能作为正式修复，
  也不能因短跑通过而标记完成。
- 中间失败：`...inductor-20260824-184945-3585505`、`...185701-3664334`。
- 通过：`...inductor-20260824-190123-3690709`。
- 2026-10-08 待复验：使用动态 metadata + `force_eager` 跑 PP8 正式 500-step；
  完成前不得标记修复通过。即使 metadata 超时消失，首步 Grad Norm 非有限也必须
  作为独立正确性问题继续定位。

## G011：编译 collective 数值不稳定

- 现象：PP/组合拓扑在 step 1 报 loss 或 gradient norm 非有限。
- 根因：当前 NPU Inductor `_c10d_functional.all_reduce` 编译路径数值不稳定。
- 处理：把该 op 加入最小 fallback，通信仍由原 HCCL process group 执行。
- 失败现场：`...inductor-20260824-185701-3664334`、
  `...193411-901195`。
- 复测：PP8、FSDP2-PP4、FSDP2-TP2-PP2 完成 10/10 steps。

## G012：8 rank 编译 worker 导致设备子进程启动超时

- 现象：`Inner_Error_Device_Subprocess_Startup_Timeout(E39007)`、507033。
- 根因：每 rank 多编译 worker 并发 `SetDevice`，与共享设备/主机资源竞争。
- 处理：`TORCHINDUCTOR_COMPILE_THREADS=1`，缓存持久复用。
- 失败：`smoke-suite-inductor-20260824-190339-3693491`。
- 复测：`fsdp2-cp4` 在 `...192959-763291` 通过。

## G013：EP 空 expert 的 grouped-mm 零核启动

- 现象：某个 expert 分组 0 token 时 CANN 计算 `coreDim=0`；queue 0 报 EE1003，
  异步路径可能 SIGSEGV。
- 处理：自定义 opaque op 对部分空组临时补零行；全空输入直接 bypass；结果只抽取
  原始行；显式 backward 按原 offsets 计算梯度。
- 多轮失败集中于 `fsdp2-tp4-ep8`，从
  `...inductor-20260824-200836-2499759` 到 `...211716-3560240`。
- 最终通过：`...inductor-20260825-164804-4064806`，step 10 loss 3.82438。

## G014：动态 pointwise 的零长度 Triton grid

- 现象：EP rank 在 step 8 接收 0 token，generated kernel 用 `xnumel=0` 启动 CANN，
  产生 EE1003/SIGSEGV。
- 根因：第一版 guard 只匹配 `_numel`，漏掉普通 `xnumel`；symbolic grouped
  autotuner 还覆盖了 base `run()`。
- 处理：匹配所有非 reduction `*numel`，同时 patch `NPUCachingAutotuner` 与
  `NPUSymbolicGroupedAutotuner`，launch 前 no-op。
- 最后一份失败：`...inductor-20260825-164021-4058637`。
- 通过：`...inductor-20260825-164804-4064806`。

## G015：grouped-mm padding offsets 被提升为 int64

- 现象：NPUGraphs DDP8 报 `Offsets have to be int32`。
- 根因：`torch.cumsum` 对整数默认提升 dtype。
- 处理：padding offsets 的 cumsum 显式指定 `dtype=torch.int32`。
- 失败：`smoke-suite-npugraphs-20260825-165012-4067511`。
- 修复后 DDP8/FSDP8 通过，后续 suite 才推进到 TP8。

## G016：NPUGraph tree 拒绝 TP `DeviceMesh` runtime inputs

- 现象：TP8 多个 rank 报 `check isinstance(inp, int) fail`。
- 诊断：AOT wrapper 有 21 个运行时 `DeviceMesh(tp=8)` 输入；不是 SymInt。
- 根因：当前 torch_npu NPUGraph tree 的 runtime input contract 不接受这些对象。
- 处理：最终 profile 从局部 unsafe-op skip 升级为
  `TORCHTITAN_NPUGRAPH_SKIP_ALL=1`；不伪称 native replay。
- 失败：`...npugraphs-20260825-165117-4069910` 以及后续定向诊断尝试。
- AOT 降级通过：`...npugraphs-20260825-170058-4085084`。

## G017：NPUGraph 降级 profile 的异步 queue 导致 TP+PP timeout

- 现象：`fsdp2-tp2-pp2` 在 step 6 后停滞；HCCL watchdog 报 COALESCED/ALLGATHER
  100 秒超时。
- 根因：replay 已禁用时仍使用 task queue 1，组合通信在该 profile 下不稳定。
- 处理：最终兼容 profile 默认 `TASK_QUEUE_ENABLE=0`；只有原生 capture 调试才显式 1。
- 失败：`smoke-suite-npugraphs-20260825-170246-4087653`。
- 通过：`...npugraphs-20260825-171724-4109299`，随后最复杂 EP 组合也通过。

## G018：共享设备 checkpoint 残余进程耗尽 stream

- 现象：最终 EP 复测在图编译前报 `EE1023 Too many streams`；每卡存在多个 checkpoint
  rank 和约 22–39 GiB HBM 占用。
- 根因：checkpoint failure-mode 场景的旧 rank 与恢复 torchrun 重叠。
- 处理：最初没有越权清理；用户明确授权后只匹配 checkpoint 模块/脚本，向 64 个
  残余进程发送 TERM，10 秒内全部退出，未使用 KILL，未删除 checkpoint 文件。
- 基础设施失败：`smoke-suite-inductor-20260825-090754-3805779`。
- 清理后最终 EP 复测通过：`...164804-4064806`。

## G019：一次验证命令类型错误

- 现象：把 shell 脚本路径误传给 `python -m py_compile`，诊断调用失败。
- 处理：shell 改用 `bash -n`，Python 仅编译 `train_npu.py`，两项均通过。
- 失败记录：`launcher-command-inductor-20260825-172508-4114206`。
- 正确 Python 复测：`...172514-4114273`；最终 graph+smoke 单测：
  `...175353-4115183`，17/17 通过。
- 该项不是图模式后端失败，但为保持完整历史仍在此记录。

## 失败保留原则

所有失败 invocation 仍在 `graph_debug_runs/`；smoke runner 将不完整 topology 改名为
`.failed-<timestamp>` 后再复测。没有为了得到“干净”结论而删除失败现场。最终判断
只读取无后缀 topology 目录中的 passed manifest。
# 2026-10-08 PP8 local metadata inference follow-up

- Dynamic PipelineStage metadata serialized cold compilation and timed out at
  rank 7. Moving metadata to the rendezvous store reproduced the same
  300-second timeout, so transport substitution was rejected.
- A one-stage-per-rank local metadata prototype removed startup metadata P2P
  and reached the first real training step without extending timeouts. It was
  subsequently rejected and removed because it bypassed PyTorch's dynamic
  protocol without using the public static `PipelineStage` constructor API.
- A separate correctness failure is now visible: step-1 loss is finite, but
  Grad Norm is NaN. Before clipping, rank 0 has 13 non-finite parameter grads
  beginning at the embedding and layer-0 MLA weights; ranks 1-7 have none.
- `TASK_QUEUE_ENABLE=0` and `2`, and zero versus uninitialized dummy hidden
  states, reproduce the same rank-0 gradient failure. Do not classify this as
  a task-queue or metadata-transport fix, and do not bypass the finite check.

# 2026-10-08 PP8 official dynamic-metadata follow-up

- The production profile keeps PyTorch dynamic metadata and its original P2P
  protocol. It only runs the original metadata probes inside the public
  `torch.compiler.set_stance("force_eager")` context.
- Server tests passed: 24 Turbo graph-compat tests and 29 experiment graph tests.
- All eight ranks completed metadata initialization and entered step 1 together;
  concurrent Ascend-Triton compilation replaced the old rank-by-rank 300-second
  metadata chain.
- The run still failed before completing step 1. Rank 7 segfaulted in
  `torch_npu.distributed._batch_isend_irecv` while exiting PyTorch's
  `_coalescing_manager` from `PipelineSchedule._batch_p2p`; elastic sent SIGTERM
  to ranks 0-6. Treat this as a separate torch_npu batched-P2P compatibility
  issue, not as evidence that the metadata fix failed or that PP8 passed.
- Timeline refinement: rank 7 owns only `norm/lm_head`, entered schedule P2P
  while transformer stages were still compiling, and failed after the wait
  window. The payload was not shown corrupt; cold-compile skew remained coupled
  to the communication critical section.
- The corrected profile captures each stage's actual inputs during the unchanged
  dynamic metadata probe, builds forward/backward caches locally before schedule
  P2P, restores RNG/buffers/grads, and synchronizes completion through the host
  rendezvous store. The 17:49 rerun completed steps 1-3 with finite Grad Norm;
  the 500-step validation remains in progress in `tmux yyb:graph`.
