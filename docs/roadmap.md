# Roadmap

This document records future architectural directions. It is not a
specification of the current implementation or public API.

## Externalized concrete components

The long-term direction is to keep the core framework generic and move
concrete observations, dynamics, encoders, trainer policies, and
model transformations into external or reference packages.

The core framework should provide:

- abstract component and transformation contracts;
- generic inference and training orchestration;
- registries for resolving concrete implementations;
- serialization and artifact metadata; and
- configuration-driven resolution of concrete components and their parameters.

Concrete implementations should be selected through symbolic names and
serializable parameters. Their implementation identity and parameters should be
captured by the reproducibility and artifact contracts.

This direction is intentionally deferred. Until the migration is explicitly
undertaken, the current bundled concrete components and runtime
`model_transformations` interface remain supported. The active behavior is
defined by the current source and the normative documents linked from the
README, especially [Design](design.md), [Training](training.md), and
[Reproducibility](reproducibility.md).
