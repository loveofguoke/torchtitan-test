# Ascend slow-device diagnostics

This diagnostic follows the MindStudio performance output lifecycle. It
separates per-device compute/HBM behavior, every two-device HCCL pair, and the
full visible-device collective.

Use at least three interleaved single-device rounds when diagnosing a suspected
slow card. Run once during known contention only as an explicitly labelled busy
baseline, then repeat while no training job is using the selected NPUs:

```bash
cd /path/to/torchtitan-test
bash tests/glm5_2_mindstudio/device_diagnostic/run_diagnostics.sh \
  --single-device-rounds 5 \
  --launch-batch 100
```

To compare the known `0,1` and `6,7` DDP behavior without loading a model, add
ordered mappings in both directions:

```bash
bash tests/glm5_2_mindstudio/device_diagnostic/run_diagnostics.sh \
  --devices 0,1,6,7 \
  --ddp-pairs '0,1;1,0;6,7;7,6' \
  --single-device-rounds 5
```

The synthetic DDP stage separately synchronizes a fixed BF16 Matmul and HCCL
AllReduce. Its per-rank rows identify whether a delay follows a physical device,
the logical rank, computation, communication, or host enqueue. It deliberately
does not include model, data loader, autograd, optimizer, or TorchTitan logic.

Rerunning the same command skips a complete generation. If the selected repeat
is incomplete or failed, it is archived automatically and retried. Use
`--repeat 2` only when a second retained measurement is intentional, and use
`--force` to replace only the selected repeat after verifying it is not active.
Relevant measurement controls are available from `--help`.

Outputs use the repository conventions:

```text
mindstudio_runs/performance/device_diagnostic/<experiment-id>/<N>-device/candidate-r1/
mindstudio_artifacts/performance/device_diagnostic/<experiment-id>/<N>-device/candidate-r1/
mindstudio_reports/performance/device_diagnostic/<experiment-id>/<N>-device/candidate-r1/
```

The run directory records `runtime.log`, `run_state.json`,
`resolved_command.json`, `resolved_launch.sh`, and the experiment overview.
Every completion, skip, and subprocess failure prints the clickable
`torchtitan-test/mindstudio_runs/performance/device_diagnostic/.../runtime.log`
path; the long-running command output is preserved there rather than relying on
terminal scrollback.
The artifact stores raw evidence under `official/`, plus a hashed `manifest.json`
and a `complete.json` marker written only after successful validation. The
report contains `summary.json` and a short human-readable verdict.

Interpretation:

- A device slow in both standalone tests points to compute, HBM, throttling,
  ECC, or resource contention.
- A high `host_enqueue_vs_median` with normal Matmul and Copy points toward
  host scheduling, CPU affinity, the framework enqueue path, or contention.
- A normal enqueue ratio but high `host_synchronized_vs_median` points toward
  device execution or queue-drain latency rather than Python launch overhead.
- Per-round raw values distinguish a device-stable regression from test-order
  drift and transient contention; do not diagnose hardware from one round.
- Normal standalone results plus slow pairs involving one physical device
  point to HCCS/HCCL or link routing.
- Normal standalone and pairwise results plus a slow full collective point to
  collective scheduling, topology, or CPU affinity.

The 10% device-throughput and 20% pair-latency deviations are triage thresholds,
not hardware pass/fail specifications.
