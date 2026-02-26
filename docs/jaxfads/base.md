# `jaxfads.base`

**Source:** `src/jaxfads/base.py`

Defines the core abstract interfaces used throughout the library:

- `Approx`: exponential-family approximate posterior family
- `Dynamics`: deterministic transition model
- `Observation`: observation/likelihood interface

See also:
- `docs/meta/notation.md` for naming/notation
- `docs/meta/design.md` for parameter-form design and conversion tables

## `Approx`

`Approx` provides conversions between:

- free (flat array for MVN; distribution-specific)
- canon (pytree)
- natural (flat array)
- moment (flat array, `E[T(z)]`)

and exposes:

- `param_size() -> int`
- `kl(moment1, moment2)`
- `sample_by_moment(key, moment, mc_size)`
- `predictive_moment(z, noise)`

## `Dynamics`

`Dynamics.forward(z, u, c)` is **deterministic**.

Process noise is owned by `XFADS` (configured via `dyn_conf.state_noise`) so
that dynamics modules remain pure transition functions.

## `Observation`

`Observation.eloglik(...)` computes the expected log-likelihood term used in the
ELBO, typically via Monte Carlo sampling from the `Approx`.
