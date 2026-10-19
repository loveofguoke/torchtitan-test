# TorchTitan feature acceptance

This directory turns the official TorchTitan feature inventory into a stable,
reviewable acceptance manifest. It does not launch training yet and it does not
change TorchTitan configuration. Its first responsibility is to prevent three
common mistakes:

1. treating an upstream config field as device acceptance;
2. hiding a hardware or backend capability gap behind a fallback;
3. mixing work that requires a TorchTitan upgrade into the current-version
   validation campaign.

Generate the deterministic JSON and Markdown views with:

```powershell
python -m tests.glm5_2_feature_acceptance.generate_matrix
```

Future launchers consume `FeatureSpec.required_topologies` and preserve the
shared smoke, precision, performance, graph, and lifecycle contracts. Runtime
evidence must update status through reviewed reports; merely regenerating this
manifest never upgrades an acceptance state.

`plans.py` now generates exact minimal smoke and profiler-off performance
commands only for features activated by topology or graph selection. A feature
that still needs a recipe/provider integration produces
`requires-integration` with no command; capability blocks likewise produce no
command. This prevents a plain baseline run from being presented as evidence
that Async TP, GraphTrainer, DeepEP, low precision, or another opt-in feature
actually ran.

`evidence.py` defines the promotion contract. Every feature/platform/run keeps
all seven gates (`function`, `numerics`, `activation`, `performance`,
`resources`, `regression`, and `reproducibility`) independently. PASS/FAIL
requires an artifact path, failures retain a summary, and `accepted=true` is
possible only when every gate has passed. Inventory status or package presence
cannot promote a feature by itself.

`optimizations.py` is the optimization registry used by A/B planning.  It
records the owning repository/runtime, platform readiness, real activation,
applicable topologies, conflicts, bottleneck hypothesis, and mandatory metrics.
An official serving optimization is kept separate from TorchTitan training;
an unavailable GLM GraphTrainer recipe remains `requires-integration` rather
than receiving a baseline command that does not activate the feature.
The generator writes both `optimization_registry.json` and
`optimization_plans.json`. The latter expands every identity by platform and
only emits candidate arguments for `runnable` entries; blocked and unfinished
integrations deliberately keep an empty candidate command.

`communication.py` converts profiler evidence into the next bounded A/B.  It
first separates exposed communication from overlapped transit, then checks
rank readiness skew, repeated FSDP all-gathers, all-to-all share, pipeline
bubbles, and effective bandwidth.  Its output is a recommendation, never an
automatic production switch.

