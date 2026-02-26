# `jaxfads.encoders`

**Source:** `src/jaxfads/encoders.py`

Neural encoders that produce flat natural-parameter updates.

- `AlphaEncoder`: per-time-step encoder `y_t -> alpha_t`
- `BetaEncoder`: backward GRU over `alpha_{T:1}` producing `beta_{1:T}`

Encoders do not depend on `Approx` directly; they only require the flat
parameter size `param_size`, injected by `XFADS`.
