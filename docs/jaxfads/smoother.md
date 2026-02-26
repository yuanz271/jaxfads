# `jaxfads.smoother`

**Source:** `src/jaxfads/smoother.py`

Implements the `XFADS` module, the main orchestrator that wires together:

- `Approx` (latent exp-family approximation)
- `Dynamics` (deterministic transition)
- `Observation` (likelihood/readout)
- encoders and filtering/smoothing routines

## Config invariants

- `conf.state_dim` and `conf.observation_dim` are the single source of truth.
  `XFADS` injects them into `dyn_conf` / `obs_conf` / encoder config.
- Encoders are `Approx`-agnostic; `XFADS` injects `param_size` computed from the
  configured `Approx`.
- Built-in subclasses are registered by importing packages for side effects:
  `jaxfads.distributions` and `jaxfads.dynamics`.
