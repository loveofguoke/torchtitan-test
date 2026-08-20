# Shared experiment primitives

`glm5_2_common` is dependency-free with respect to individual experiments. It
owns canonical accelerator selection, topology definitions, topology selection,
and conflict-checked execution feature composition.

Dependency direction is one way:

```text
glm5_2_common
  <- precision <- checkpoint / stability
  <- performance
  <- precision + performance <- combination <- graph convenience entry points
```

Experiment modules must not be imported by `glm5_2_common`. Features contribute
arguments and environment variables through `TrainingFeature`; only the central
combination workflow assembles them into a training command.

## Standard experiment lifecycle

Training experiments use one reproducible input contract whenever numerical
results or restart behavior are compared:

1. `--data` creates a step-0 model checkpoint and topology-independent fixed
   token plan.
2. `--capture ...` or the experiment's training action consumes that fixture.
3. `--compare` reads portable artifacts and writes the report without an
   accelerator.

The parity, formal precision, combination/graph, checkpoint, and stability
experiments follow this contract. A fixture can be generated on either backend
and is reused across topologies when its training settings are unchanged.

The standalone performance profiler is the deliberate exception. It measures
profiling overhead and runtime behavior and does not claim numerical
comparability. Use the combination runner when fixed inputs, graph mode,
distributed execution, precision comparison, and profiling must be enabled in
the same training process.
