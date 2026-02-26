# `jaxfads.observations`

**Source:** `src/jaxfads/observations.py`

Observation models / likelihoods.

Key components:

- `GLM`: wraps a readout + likelihood
- Likelihoods: `Poisson`, `Gaussian`
- Readout initializers: e.g. `fa` initializer for Factor Analysis loading matrix

The likelihoods consume the latent `Approx` through the `Observation.eloglik`
API.
