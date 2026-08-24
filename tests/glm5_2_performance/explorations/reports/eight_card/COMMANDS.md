# Eight-card report and analysis commands

Every training run keeps its exact invocation in
`explorations/runs/<run-name>/command_history.jsonl`; every generated
`torchrun` argv is in the adjacent `manifest.json`. The commands below cover
the cross-run report and manual evidence checks used by the narrative analysis.

## Generate the eight-card table and charts

```bash
cd /workspace/y50064852_yyb/torchtitan-test
python tests/glm5_2_performance/explorations/tools/summarize_topologies.py \
  --steps 20 --latest-replicate --world-size 8 \
  --markdown tests/glm5_2_performance/explorations/reports/eight_card/EIGHT_CARD_TOPOLOGY_REPORT.md \
  --html performance_reports/eight_card/npu-eight-card-topology-comparison.html
```

The generator also appends the resolved Python executable, cwd, argv, outputs,
and timestamp to `explorations/history/report_generation_commands.jsonl`.

## Inspect DDP8 structured attribution

```bash
run=tests/glm5_2_performance/explorations/runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce
jq '{comparison:.profile_phases.comparison,ranks:.distributed_step_trace.ranks,comm:.communication_summary.rows}' "$run/analysis.json"
```

## Verify official analysis return codes

```bash
run=tests/glm5_2_performance/explorations/runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce
for file in "$run"/tool_commands/*.json; do
  jq -e '.return_code == 0' "$file"
done
```

## Count official host-bound classifications

```bash
base=performance_runs/npu-ddp8-bf16-s20-l8-b64-seq128-seed61-distributed-r1-488247ce
files="$base"/communication_bottleneck_rank_*/cluster_analysis_output/CommunicatonBottleneckAnalysis/communication_bottleneck.csv
wc -l $files
rg -o 'Host-bound|Network-bound|GroupedMmBackward0|MatmulBackward0|AddBackward0|ToCopyBackward0|FSDP::pre_forward|aten::empty|aten::add_|aten::as_strided|Event::record|Event::wait|c10d::allreduce_' $files \
  | sed 's/.*://' | sort | uniq -c | sort -nr
```

Each of the eight CSVs has one header plus 20 result rows. The count therefore
contains 160 classifications, all Host-bound.

## Unit test the report selector and stability derivation

```bash
python -m unittest \
  tests.unit_tests.test_glm5_2_performance_explorations -v
```

