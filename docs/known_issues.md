# Known Issues

## Multi-device sharded `train()` silently computes a wrong loss (unresolved)

**Status:** confirmed, root cause not yet isolated inside JAX/XLA's partitioner; workaround available.

**Symptom:** when more than one JAX device is visible (no `CUDA_VISIBLE_DEVICES`
pinning) and `train()`/`_run_training_loop`'s default `data_sharding` /
`model_sharding` (`eqx.filter_shard` + `jax.sharding.NamedSharding` over a
`jax.make_mesh((n_devices,), ("batch",))` mesh, `PartitionSpec("batch")` for
data / `PartitionSpec()` for the model) is active, the training loss is
**numerically wrong**, not just differently-conditioned. This was found while
debugging a NaN divergence in an HBN downstream project's jaxfads integration
(the eeg repo's `scripts/python/jaxfads_hbn/`), on a config with `k=12`,
`latent=12`, `p=2`, `hidden_size=160`, `mc_size=4`, batch size 512, on 8
visible CUDA devices.

**Minimal reproduction** (isolates the bug down to a single `jnp.mean` call,
independent of the model):

```python
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import sharding as jshd

n_devices = len(jax.devices())  # 8 in the reproducing environment
mesh = jax.make_mesh((n_devices,), ("batch",))
data_sharding = jshd.NamedSharding(mesh, jshd.PartitionSpec("batch"))

x = jnp.arange(512, dtype=jnp.float32)
x_sharded = eqx.filter_shard(x, data_sharding)

@eqx.filter_jit
def trivial_mean(x):
    return jnp.mean(x)

print(float(trivial_mean(x_sharded)))  # got 15.875, want 255.5
```

The correct global mean of `arange(512)` is `255.5`. Under this exact
sharding + `eqx.filter_jit` combination, it instead returns `15.875` — not a
rounding difference, a wrong answer to a trivial reduction.

**Isolation on the real model** (same step-0 batch/model/key throughout,
`jaxfads_hbn` config `k=12 latent=12 p=2 hidden_size=160 mc_size=4`,
`batch_size=512`, `n_trials=20000` from the HBN FunwithFractals movie data,
seed `20250417`):

| variant | loss |
|---|---|
| (a) fully unsharded `batch_loss`, standalone `eqx.filter_jit` | 9.327450 |
| (b) sharded inputs, standalone `eqx.filter_jit` (no `value_and_grad`) | 14.304255 |
| (c) sharded inputs inside `eqx.filter_value_and_grad` (matches `_run_training_loop`'s real `train_step`) | 14.304255 |

(a) vs (b) differ despite (b) computing no gradient at all -- proving the
discrepancy is a forward-pass issue (the `jnp.mean(free_energy)` inside
`batch_loss`), not something introduced by reverse-mode AD. (b) == (c) shows
`filter_value_and_grad` doesn't add or remove any further discrepancy once
sharding is already present.

**Consequence:** any `train()` call made with more than one JAX device visible
and no explicit device pinning trains against a silently wrong loss/ELBO. This
most plausibly explains downstream instabilities that don't reproduce on a
single device or a small subsample; a systematically wrong (and likely
inconsistently-wrong-per-term, since different reduction shapes may partition
differently) loss estimate would distort the relative balance between
reconstruction-fit and KL/process-noise terms.

**Workaround (in use now):** pin the process to a single visible GPU via
`CUDA_VISIBLE_DEVICES=<id>` before importing `jax`. This collapses
`len(jax.devices())` to 1, so the mesh is trivial and no cross-device
reduction is needed -- correctness no longer depends on the buggy code path.
For an embarrassingly-parallel multi-run campaign (e.g. many seeds/configs),
round-robin one GPU per process rather than sharding one process across
several GPUs.

**Not yet done:**
- Root-cause why GSPMD's auto-partitioner isn't inserting a correct
  cross-shard collective for this `jnp.mean` (and, by extension, for
  `batch_elbo`'s reductions) under this mesh/`PartitionSpec` combination and
  JAX/jaxlib version -- vs. some other sharding API (e.g. explicit
  `shard_map`, or a different `PartitionSpec` layout) not being subject to it.
- Audit whether *every* prior multi-GPU `train()` run in this repo and in
  downstream projects (any run launched without `CUDA_VISIBLE_DEVICES`
  pinning on a multi-GPU host) used a silently wrong loss. In the eeg
  downstream project specifically, `scripts/shell/run_jaxfads_hbn.sh` (the
  CCD-oriented single-run launcher) does not pin a device by default and
  needs this fix; a round-robin multi-seed launcher in that project already
  pins one GPU per process and was unaffected.
