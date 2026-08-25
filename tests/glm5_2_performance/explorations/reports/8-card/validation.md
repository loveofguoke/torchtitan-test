# Eight-card evidence validation

## Result

The profiler-active attribution matrix is complete for all 13 declared
eight-rank topologies. The selected successful run for every topology has:

- eight parsed Ascend profiler rank roots;
- 11 official analysis invocations: cluster, cluster time summary, free
  analysis, and communication bottleneck for ranks 0-7;
- zero non-zero official-tool return codes;
- `manifest.json`, `metrics.jsonl`, `analysis.json`, `artifacts.json`,
  `command_history.jsonl`, and a readable `readme.md`;
- a generated HTML performance report and a MindStudio-importable trace root.

Profiler-off screening is independently complete for the same 13 topologies.
The profiler-off rows remain the throughput authority; profiler-active rows
are attribution evidence only.

## Selected attribution runs

| Topology | Selected run | Roots | Official tools | Failed tools | HTML |
| --- | --- | ---: | ---: | ---: | --- |
| cp8 | `npu-cp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-03d875ae` | 8 | 11 | 0 | yes |
| ddp8 | `npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce` | 8 | 11 | 0 | yes |
| ep8 | `npu-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r4-ccb6f22c` | 8 | 11 | 0 | yes |
| fsdp2-cp4 | `npu-fsdp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-60aa6993` | 8 | 11 | 0 | yes |
| fsdp2-pp4 | `npu-fsdp2-pp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-dde8b599` | 8 | 11 | 0 | yes |
| fsdp2-tp2-pp2 | `npu-fsdp2-tp2-pp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-02a4a78f` | 8 | 11 | 0 | yes |
| fsdp2-tp4 | `npu-fsdp2-tp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-7088c3b5` | 8 | 11 | 0 | yes |
| fsdp2-tp4-ep8 | `npu-fsdp2-tp4-ep8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-e6d2b2bd` | 8 | 11 | 0 | yes |
| fsdp4-tp2 | `npu-fsdp4-tp2-bf16-s20-l8-b64-seq128-seed61-distributed-r2-a955eda9` | 8 | 11 | 0 | yes |
| fsdp8 | `npu-fsdp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-8501ffd6` | 8 | 11 | 0 | yes |
| pp8 | `npu-pp8-bf16-s20-l8-b64-seq128-seed61-distributed-r2-462adeae` | 8 | 11 | 0 | yes |
| tp2-cp4 | `npu-tp2-cp4-bf16-s20-l8-b64-seq128-seed61-distributed-r2-ddc3f9b1` | 8 | 11 | 0 | yes |
| tp8 | `npu-tp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-dba2190c` | 8 | 11 | 0 | yes |

## Validation method

The validation selects the unique profiler-active run referenced by the
generated attribution table, resolves its container paths back to the host
workspace, and checks rank roots, tool JSON return codes, and HTML existence.
Container paths begin with `/workspace/y50064852_yyb/torchtitan-test`; the
corresponding host prefix is `/home/y50064852_yyb/torchtitan-test`. Checking a
container path directly from the host produces a false missing-file result and
is not evidence loss.

The exact selected run and evidence links remain in
[comparison.md](comparison.md). Failed or superseded attempts are retained in
their own run directories and are not used for this success matrix.
