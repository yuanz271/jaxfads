# `jaxfads.distributions.mvn`

**Source:** `src/jaxfads/distributions/mvn.py`

Multivariate normal (`MVN`) exponential-family approximation.

## Structures

`MVN(dim, structure=...)` supports:

- `structure="full"`:
  - natural size: `D + D^2`
  - moment size: `D + D^2` with `T2(z) = -1/2 zz^T`
- `structure="diag"`:
  - natural size: `2D`
  - moment size: `2D` with `T2(z) = -1/2 (z ⊙ z)`

Invariant for callers:
- `MVN.unpack(moment)` returns `(mean, cov)` where `cov` is **always** full
  `(D, D)` (diagonal-valued in diag mode).

See `docs/meta/design.md` and `docs/meta/notation.md` for parameter-form
conventions.
