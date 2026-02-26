# `jaxfads.core`

**Source:** `src/jaxfads/core.py`

Filtering/smoothing primitives.

Key functions:

- `filter(...)`: forward variational filtering
- `bismooth(...)`: bidirectional smoothing (if enabled)
- `expected_predictive_moment(...)`: Monte Carlo estimate of Eq. (12)

See `docs/meta/algorithm.md` for the math overview and mapping to code.
