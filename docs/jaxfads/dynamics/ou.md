# `jaxfads.dynamics.ou`

**Source:** `src/jaxfads/dynamics/ou.py`

Built-in Ornstein–Uhlenbeck (OU) drift dynamics.

`OUDynamics` implements the zero-mean OU drift with Euler discretization:

```
z_next = z + dt * (-theta * z)
```

Notes:
- `theta` is trainable (stored as an unconstrained parameter and constrained to
  be positive).
- `dt` is a constant step size from `dyn_conf`.
- Process noise is owned by `XFADS` (`dyn_conf.state_noise`).
