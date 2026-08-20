# Shared experiment primitives

`glm5_2_common` is dependency-free with respect to individual experiments. It
owns canonical accelerator selection, topology definitions, topology selection,
and conflict-checked execution feature composition.

Dependency direction is one way:

```text
glm5_2_common
  <- precision / performance / checkpoint / stability
  <- combination
       <- graph convenience entry points
```

Experiment modules must not be imported by `glm5_2_common`. Features contribute
arguments and environment variables through `TrainingFeature`; only the central
combination workflow assembles them into a training command.
