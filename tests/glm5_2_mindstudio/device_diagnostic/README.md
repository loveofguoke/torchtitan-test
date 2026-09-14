# Ascend slow-device diagnostics

This diagnostic follows the MindStudio performance output lifecycle. It
separates per-device compute/HBM behavior, every two-device HCCL pair, and the
full visible-device collective.

Run only while no training job is using the selected NPUs:

```bash
cd /path/to/torchtitan-test
bash tests/glm5_2_mindstudio/device_diagnostic/run_diagnostics.sh
```

Use `--repeat 2` to retain another run of the same configuration, or `--force`
to replace the selected repeat after verifying that it is not active. Relevant
measurement controls are available from `--help`.

Outputs use the repository conventions:

```text
mindstudio_runs/performance/device_diagnostic/<experiment-id>/<N>-device/candidate-r1/
mindstudio_artifacts/performance/device_diagnostic/<experiment-id>/<N>-device/candidate-r1/
mindstudio_reports/performance/device_diagnostic/<experiment-id>/<N>-device/candidate-r1/
```

The run directory records `runtime.log`, `run_state.json`,
`resolved_command.json`, `resolved_launch.sh`, and the experiment overview.
The artifact stores raw evidence under `official/`, plus a hashed `manifest.json`
and a `complete.json` marker written only after successful validation. The
report contains `summary.json` and a short human-readable verdict.

Interpretation:

- A device slow in both standalone tests points to compute, HBM, throttling,
  ECC, or resource contention.
- Normal standalone results plus slow pairs involving one physical device
  point to HCCS/HCCL or link routing.
- Normal standalone and pairwise results plus a slow full collective point to
  collective scheduling, topology, or CPU affinity.

The 10% device-throughput and 20% pair-latency deviations are triage thresholds,
not hardware pass/fail specifications.
