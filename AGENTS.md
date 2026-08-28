# Agent instructions

Read `DEPENDENCY_AUDIT.md` before changing any experiment framework code.

- Keep model mathematics in TorchTitan and NPU patches in TorchTitanTurbo.
- Audit every consumer in the dependency map after changing common topology,
  naming, execution, lifecycle, graph, profiler, fixture, or report code.
- Preserve the documented force/resume generation contract and add regression
  tests for every lifecycle change.
- Keep experiment defaults backward compatible unless the user explicitly
  requests a new experiment identity.
- Do not commit unless the user explicitly requests a commit.
