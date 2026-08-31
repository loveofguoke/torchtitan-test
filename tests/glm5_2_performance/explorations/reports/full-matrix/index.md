# all-preset 全拓扑矩阵

## 当前状态

状态：`running`（2026-08-31 09:59 CST）。原 8 卡任务已由用户停止，遗留在 NPU2 的单个
孤儿 rank 经精确核对后清理。首个 `single/overview` capture 与 CANN 解析已成功；advisor
因图环境未透传隔离 CLI 路径而 fail-fast。兼容修复和 48 项单测通过后，队列已按 manifest
恢复，成功 capture 不会重复训练。完整故障与恢复命令见命令账本第 9 节。

## 已验证的首个成员

`single/overview`（run `npu-single-bf16-s30-l8-b64-seq128-seed61-overview-r31-92543cf1`）
已形成 capture manifest、30 条训练指标、CANN 解析数据、advisor HTML、综合 HTML 和探索
run 文档。训练侧中位 step time 为 3.052 秒，吞吐中位数为 2,684.54 tok/s，峰值 active/
reserved HBM 为 1.323/1.400 GiB，alloc retry/OOM 均为 0。

overview 活跃窗口为 steps 13–17。advisor 将该窗口拆为 20,032.682 ms E2E、4,877.765 ms
计算、0 ms 未掩盖通信和 15,154.917 ms idle。这里的 idle 占比是定位线索，不是最终稳态
吞吐结论：该 run 使用 profiler-active，且同步解析导致 step 18 额外停顿 146.07 秒；正式
speedup 仍必须引用 profiler-off 组合矩阵。

overview 没有采集 op summary，因此 advisor 明确跳过 dynamic-shape、AICore bound、AiCPU
和部分融合检查；这些问题由后续 `operator`/`kernel` preset 回答。advisor 文本还把运行环境
显示为 CANN 8.0/torch 2.1，与 launcher 记录的 CANN 9.1/torch 2.14 不一致，属于当前
msprof-analyze 规则元数据边界，不能用该显示值替代三仓环境报告。

第二个运行期问题出现在原始 `tp8/overview`：rank0 同步 CANN 解析时，其他 rank 在
collective 等待超过 TorchTitan 的 100 秒训练 timeout，导致 rank2 SIGABRT。该失败保留为
失败证据。workflow 现对所有多 rank NPU 同步 preset 使用训练后 offline parse；TP8 定向
复验 30/30 steps、3 分 11 秒离线解析、advisor 和 HTML 均通过。这个耗时本身证明解析不能
安全放在 live distributed step 中。

计划范围为 27 个 1/2/4/8 卡实验拓扑 × 8 个非冗余 profiler preset，共 216 次独立
capture。每项使用 30 steps、skip 10、warmup 2、active 5、batch 8/64、sequence 128、
seed 61 和 `replicate=31`。

完整执行过程、精确命令、分析工具展开和输出 contract 见
[all-preset 命令账本](../../history/all-preset-commands.md)。

## 完成判定

本页只有同时满足以下条件才改为 `complete`：

1. 216 个完整实验身份均有 capture manifest；
2. 每个 run 的训练指标和 profiler window 完整；
3. offline preset 完成 torch_npu 离线解析；
4. advisor 均有返回状态；
5. 多 rank capture 完成 cluster、cluster time、free analysis 和逐 rank communication
   bottleneck；
6. 每个 run 的 HTML 与 `explorations/runs/.../readme.md` 链接可达；
7. 汇总明确区分 profiler-off 历史吞吐和 profiler-active 归因，不用 active trace 计算正式
   speedup；
8. 失败项保留原始路径、首次错误、可恢复命令和状态，不从分母中静默删除。

## 待生成分析

矩阵完成后，本目录将补充：

- preset × topology 完整覆盖表；
- 1/2/4/8 卡同 rank 综合分析；
- 计算、通信、内存、host/runtime、system 五类瓶颈地图；
- 与 eager/Inductor 60-run A/B 的关联分析；
- 每个结论的证据文件、工具命令、置信度和潜在优化注入位置；
- 面试版问题—证据—判断—验证闭环。

排队本身不是失败，也不是实验通过。权威状态来自 queue log、run manifest 和 suite report。
