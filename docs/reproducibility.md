# Reproducibility

This document defines general principles for reproducible model fits and saved
artifacts.

## Scope and assumptions

A reproducible run is determined by:

1. an exact snapshot of the codebase;
2. a fixed machine and runtime setting; and
3. a fully resolved configuration, together with the specified input data and
   preprocessing.

The codebase snapshot and machine/runtime setting are assumptions of the
contract. Every other choice that can affect a fit must be explicit in the
resolved configuration or in the self-contained artifact metadata defined
below.

The machine/runtime setting includes the hardware, device allocation, backend,
numerical precision, dependency environment, and relevant runtime settings.
Under fixed conditions, repeated runs must agree within a stated numerical
tolerance. Bitwise-identical output is a stronger implementation-specific
claim and is not required unless explicitly tested.

## Configuration completeness

Every model and training choice that can affect a fit must be representable in a
serializable configuration. This includes, as applicable:

- model dimensions, architecture, and component choices;
- inference and approximation settings;
- optimizer type and hyperparameters;
- schedules, regularization, and parameter-freezing policy;
- closed-form or non-gradient update policies;
- update ordering and policy parameters;
- random seeds;
- data splitting, ordering, and shuffling; and
- preprocessing, masking, normalization, units, and external-resource
  identifiers.

A fit-affecting choice must not be hidden in an undocumented default, a global
variable, or an unrecorded runtime object.

## Configuration resolution

A user-provided configuration must be resolved against the built-in default
configuration before model construction or training. Resolution must produce a
complete, explicit, serializable configuration.

The resolved configuration—not only the partial user configuration—is the
reproducibility record and must be stored with every model save and checkpoint.
Built-in defaults are part of the behavior: changing a default can change the
effective fit even when the user configuration is unchanged.

## Randomness

All stochastic behavior must derive from explicit seeds and deterministic random
state management. The reproducibility record must identify every independently
controlled source of randomness, including initialization, data ordering,
augmentation, masking, sampling, and stochastic optimization.

Ambient process-global random state must not silently affect a reproducible fit.
A resumable checkpoint must preserve the random state required for continuation.

## Data and preprocessing

The reproducibility record must identify the exact input data and preprocessing
used by a fit. It must specify, as applicable:

- dataset version, path, or content hash;
- train/validation/test split and split seed;
- ordering and shuffling policy;
- filtering, masking, normalization, imputation, and unit conventions; and
- preprocessing implementation and version.

A path or hash identifies external data but does not make an artifact
self-contained. If continuation from the artifact alone is required, the data
and preprocessing state—or an embedded data snapshot—must also be available
from the artifact.

## Extension points

An extension that changes a fit must have a reproducible representation. It must
be one of:

1. a built-in policy selected by serializable configuration fields;
2. a registered symbolic name with serializable parameters; or
3. an implementation and provenance manifest identifying the exact extension
   code and its parameters.

This applies to user-defined optimizers, regularizers, schedules, inference
policies, and non-gradient transformations. An arbitrary runtime callable
cannot support a configuration-only reproducibility guarantee unless its
implementation is captured by a manifest or included in the artifact.

## Model saves

A model save is a self-contained inference artifact. It must contain everything
needed to reconstruct and evaluate the trained model without the original
interactive session or configuration object, including:

- trained model state;
- static model and inference configuration;
- the fully resolved configuration;
- serialization-format and schema version;
- identifiers, parameters, and provenance for custom components; and
- data and preprocessing metadata needed to interpret model inputs.

Loading a model save must reconstruct the same model semantics. Re-saving a
loaded artifact should preserve its resolved configuration and model state up
to the serialization format's numerical tolerance.

## Training checkpoints

A training checkpoint is a strict superset of a model save. In addition to all
model-save contents, it must contain the state required to continue training:

- optimizer state;
- epoch, step, and schedule state;
- current random state;
- ordered policy configuration and any policy state;
- data iteration, split, and shuffling state; and
- the data and preprocessing snapshot if continuation must work from the
  checkpoint alone.

Loading a checkpoint must reconstruct both the model state and the training
state. Continuing from it must reproduce the same subsequent trajectory as an
uninterrupted run under the same codebase and machine/runtime setting.

A model save is sufficient for inference and model reconstruction. It is not
sufficient for exact training continuation unless it also contains the
additional checkpoint state above.

## Guarantees

Under an exact codebase snapshot and fixed machine/runtime setting:

- the same resolved configuration and specified data produce the same fit
  within numerical tolerance;
- a model save is independently loadable and semantically equivalent to the
  model that produced it; and
- a checkpoint is independently loadable and can reproduce the continuation
  of the training run.

“Same” means matching resolved configurations, model state, predictions,
losses, and relevant metrics within the stated tolerance. It does not require
byte-identical archive contents when metadata, ordering, or compression may
differ.

## Verification requirements

Reproducibility must be tested rather than assumed. The verification suite
should cover:

1. repeated fits from the same resolved configuration and data;
2. model-save/load equivalence;
3. checkpoint-load equivalence;
4. checkpoint resume versus uninterrupted training; and
5. matching predictions, losses, and relevant evaluation metrics.

Tests must record the backend, numerical precision, tolerance, and
code/dependency setting under which the guarantee is asserted.
