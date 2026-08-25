# 8-card 图模式实验

默认使用：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  tests/glm5_2_graph_debug/run_graph_mode.sh BACKEND smoke \
  --topology TOPOLOGY
```

## 正式结果

| topology | Inductor 训练调用 | NPUGraphs profile 训练调用 |
|---|---|---|
| `ddp8` | `...inductor-20260824-181711-1826722` | `...npugraphs-20260825-165117-4069910` |
| `fsdp8` | `...inductor-20260824-181711-1826722` | `...npugraphs-20260825-165117-4069910` |
| `tp8` | `...inductor-20260824-182622-1906926` | `...npugraphs-20260825-170058-4085084` |
| `cp8` | `...inductor-20260824-183751-3070299` | `...npugraphs-20260825-170246-4087653` |
| `pp8` | `...inductor-20260824-190123-3690709` | `...npugraphs-20260825-170246-4087653` |
| `ep8` | `...inductor-20260824-190339-3693491` | `...npugraphs-20260825-170246-4087653` |
| `fsdp2-tp4` | `...inductor-20260824-190339-3693491` | `...npugraphs-20260825-170246-4087653` |
| `fsdp2-cp4` | `...inductor-20260824-192959-763291` | `...npugraphs-20260825-170246-4087653` |
| `tp2-cp4` | `...inductor-20260824-193411-901195` | `...npugraphs-20260825-170246-4087653` |
| `fsdp4-tp2` | `...inductor-20260824-193411-901195` | `...npugraphs-20260825-170246-4087653` |
| `fsdp2-pp4` | `...inductor-20260824-195036-2017092` | `...npugraphs-20260825-170246-4087653` |
| `fsdp2-tp2-pp2` | `...inductor-20260824-200530-2497225` | `...npugraphs-20260825-171724-4109299` |
| `fsdp2-tp4-ep8` | `...inductor-20260825-164804-4064806` | `...npugraphs-20260825-172013-4111556` |

表中省略号统一展开为 `graph_debug_runs/smoke-suite-<backend>-<stamp>/reports/report.md`。
一次 suite 调用可能先通过若干 topology、再在下一个 topology 失败，因此调用级报告
可能整体标为 FAILED。每一行的最终状态必须读取：

```text
smoke_runs/npu-glm5_debugmodel-s10-b64-seq128-seed61-<backend>-model/
  <topology>/manifest.json
```

上述 26 个正式 manifest 均为 `status: passed`、`return_code: 0`；加上 single 和
ddp2 后，两个后端 profile 各 15/15。

## 关键路径

- TP8 验证 `aten.complex` DTensor strategy。
- PP8/FSDP2-PP4 验证普通 metadata P2P、HCCL 建链超时和 collective fallback。
- FSDP2-TP2-PP2 验证组合通信在 task queue 0 下不再 watchdog timeout。
- FSDP2-TP4-EP8 验证空 expert、int32 offsets 和零 numel Triton guard；Inductor
  最终 loss 3.82438，NPUGraphs AOT profile 最终 loss 3.82464。
