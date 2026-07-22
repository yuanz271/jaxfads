# Known Issues

## Multi-device sharded `train()` silently computes a wrong loss (root cause: machine-specific hardware, not a general JAX/XLA bug)

**Status:** root-caused to a magnesium-specific hardware fault (elevated NVRM
error history on the NUMA-node-1 PCIe domain serving GPUs 4-7), not a general
JAX/XLA/equinox/jaxfads defect. A **separate**, likely-general 2-device hang
bug was also found during the investigation (see below). Workaround (single-
GPU pinning) remains in place and is sufficient for both.

**Symptom:** when more than one JAX device is visible (no `CUDA_VISIBLE_DEVICES`
pinning) and `train()`/`_run_training_loop`'s default `data_sharding` /
`model_sharding` (`eqx.filter_shard` + `jax.sharding.NamedSharding` over a
`jax.make_mesh((n_devices,), ("batch",))` mesh, `PartitionSpec("batch")` for
data / `PartitionSpec()` for the model) is active, the training loss can be
**numerically wrong**, not just differently-conditioned. This was found while
debugging a NaN divergence in an HBN downstream project's jaxfads integration
(the eeg repo's `scripts/python/jaxfads_hbn/`), on a config with `k=12`,
`latent=12`, `p=2`, `hidden_size=160`, `mc_size=4`, batch size 512, on 8
visible CUDA devices, on host **magnesium**.

**Repro scripts:** `scripts/gpu_sharding_repro/` (this repo) --
`fresh_process_sweep.sh` (primary repro; one-shot fresh-process-per-config
device/topology sweep with timing), `shard_identity_probe.py` (identifies
*which* shard's contribution went missing, to distinguish position-based
from device-based causes), `per_device_sanity.py` (single-GPU compute
correctness, no sharding), `minimal_repro.py` (parametrized single-process
repro; note its `--n-trials` loop mode reuses one compiled executable across
calls and was observed to mask the bug much more often than fresh-process
invocation -- prefer `fresh_process_sweep.sh` for reproduction attempts).
See `scripts/gpu_sharding_repro/README.md` and `results/*.log` for full
per-machine run logs.

### Two distinct failure modes were found, not one

1. **n=2 device meshes hang** (near-deterministic, not just occasional).
   Reproduced on both magnesium and palladium (identical hardware/software
   otherwise), independent of which physical GPU pair, independent of
   cold-vs-warm process position (tested by running the same pair first and
   last in a sweep -- it hung both times, ruling out a cold-start artifact).
   This looks like a genuine, likely host-independent JAX/XLA collective
   deadlock specific to exactly 2 participants, separate from the
   wrong-answer failure mode below.

2. **n=4 / n=8 device meshes probabilistically compute a wrong reduction**
   (one or more shards' contributions silently dropped from the collective
   sum; divisor unchanged). This mode is **machine-specific**: reproduced
   repeatedly on magnesium (~9/70 fresh-process trials, ~13%), never
   reproduced on palladium (0/28 fresh-process trials) under otherwise
   identical hardware, driver, and JAX/jaxlib/equinox versions.

### Minimal reproduction (isolates the bug down to a single `jnp.mean` call,
independent of the model)

```python
import equinox as eqx
import jax
from jax import numpy as jnp
from jax import sharding as jshd

n_devices = len(jax.devices())
mesh = jax.make_mesh((n_devices,), ("batch",))
data_sharding = jshd.NamedSharding(mesh, jshd.PartitionSpec("batch"))

x = jnp.arange(512, dtype=jnp.float32)
x_sharded = eqx.filter_shard(x, data_sharding)

@eqx.filter_jit
def trivial_mean(x):
    return jnp.mean(x)

print(float(trivial_mean(x_sharded)))  # wrong on a bad run, e.g. 15.875 or 203.5625; want 255.5
```

**Trigger condition isolated** (see `scripts/gpu_sharding_repro/minimal_repro.py`
control-condition flags `--no-donate` / `--no-alias-output`): the failure
requires **both** (a) `eqx.filter_jit(..., donate="all")` (or equivalent
`donate_argnums`) **and** (b) the jitted function's output aliasing/re-
returning a donated, sharded input that also feeds a cross-device reduction.
Removing either condition was never observed to fail in hundreds of trials.
This exactly matches the real `_run_training_loop`'s `train_step`: `model`/
`opt_state` are donated, participate in a sharded forward pass with
reductions, and the updated `model`/`opt_state` (reusing the donated memory)
are returned every step.

Plain `jax.jit` + `jax.device_put` with the identical sharding, and
`eqx.filter_shard` used eagerly outside jit, were both verified correct in
isolation -- `eqx.filter_shard`/`eqx.filter_jit` are thin, correct wrappers;
they are the vehicle that exposes the defect, not its source.

**Isolation on the real model** (same step-0 batch/model/key throughout,
`jaxfads_hbn` config `k=12 latent=12 p=2 hidden_size=160 mc_size=4`,
`batch_size=512`, `n_trials=20000` from the HBN FunwithFractals movie data,
seed `20250417`, on magnesium):

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

### Root-cause investigation: machine-specific hardware, not general JAX/XLA

**Cross-machine testing.** The same repro scripts were run on 3 hosts with
identical GPU model (8x NVIDIA L40S), topology (two NUMA nodes of 4 GPUs
each, PCIe/QPI-connected, no NVLink), driver (575.57.08), and JAX stack
(jax/jaxlib 0.6.2, equinox 0.13.8):

- **magnesium** (original discovery host): wrong-answer bug reproduces
  repeatedly, concentrated on device combinations including GPUs 4-7 (the
  second NUMA node). n=2 hang also reproduces.
- **palladium**: wrong-answer bug **never** reproduced (0/28 fresh trials
  across multiple sessions, idle and otherwise-clean conditions). n=2 hang
  **does** reproduce here too (same as magnesium), including a controlled
  cold-first/warm-last test ruling out a process-position artifact.
- **iodine**: testing paused early -- one GPU had a large resident
  allocation from another process at the time, confounding any rate
  estimate; not used for conclusions.

**Position-vs-device discriminator.** On magnesium, permuting the
`CUDA_VISIBLE_DEVICES` order of the same 4 physical GPUs showed that when a
failure occurred, the *same* failure signature (e.g. "only mesh position 0's
shard survives, all others dropped") recurred with **different physical
GPUs** occupying that position across separate trials. This argues the
failure tracks **mesh position / collective structure**, not a single
defective GPU die -- consistent with a shared-infrastructure fault (PCIe
switch/riser/root-complex) rather than one bad chip. A `shard_identity_probe.py`
run also showed multi-shard dropouts (not always exactly one shard), further
supporting a timing race rather than a fixed per-device corruption.

**Confirmed hardware fault history on magnesium.** `journalctl -k` (prior
boot) showed, at **2026-06-15 22:42:xx**, repeated NVRM errors on GPU 6
(PCI `0000:c1:00.0`, minor number 5):

```
NVRM: kflcnWaitForHalt_TU102: Timeout waiting for Falcon to halt
NVRM: gpuWaitForGfwBootComplete_TU102: GSP failed to halt with GFW_BOOT: (progress 0xff)
NVRM: kgspWaitForGfwBootOk_TU102: failed to wait for GFW boot complete: 0x65 VBIOS version 95.02.66.00.15
NVRM: kgspWaitForGfwBootOk_TU102: (the GPU may be in a bad state and may need to be reset)
NVRM: RmInitAdapter: Cannot initialize GSP firmware RM
NVRM: GPU 0000:c1:00.0: RmInitAdapter failed! (0x62:0x65:1941)
```

This is a documented NVIDIA GSP-firmware boot-timeout failure mode (distinct
from the "bad register read"/`0xbadf...` sentinel pattern also seen in the
same window, itself a known "GPU fell off the bus" signature --
see [NVIDIA/open-gpu-kernel-modules#688](https://github.com/NVIDIA/open-gpu-kernel-modules/issues/688)
and NVIDIA's own developer-forum guidance on deciphering NVRM/Xid messages).
The machine was rebooted at 2026-06-15 23:38:23 (`uptime -s`), immediately
after this failure window, and has not disappeared again since. Palladium's
boot time (2026-05-22 20:50:06) shows no corresponding incident.

**NVRM error tally since 2026-06-01** (`journalctl -k` across all boots,
grouped by GPU PCI address), on magnesium:

| GPU | PCI addr | NUMA node | NVRM error mentions |
|---|---|---|---|
| 0 | 01:00.0 | 0 | 2 |
| 1 | 21:00.0 | 0 | 2 |
| 2 | 41:00.0 | 0 | 4 |
| 3 | 61:00.0 | 0 | 0 |
| 4 | 81:00.0 | 1 | 8 |
| 5 | a1:00.0 | 1 | 8 |
| 6 | c1:00.0 | 1 | 6 |
| 7 | e1:00.0 | 1 | 4 |

NUMA0 total: 8. NUMA1 total: 26 (~3x). GPU 6 is not uniquely bad (GPUs 4 and
5 have equal-or-higher counts) -- the elevated error rate is a **NUMA1-group
property**, not a single-card property. This matches the wrong-answer bug's
concentration on device combinations drawn from 4-7, and the
position-independence finding above (a shared riser/PCIe-switch/root-complex
fault on that NUMA node, rather than one defective die, would affect
whichever GPU happens to occupy the "victim" position in a given trial).
Live health counters (ECC, temperature, power, clock-throttle reasons) for
GPU 6 currently show no anomaly relative to siblings -- the fault is
intermittent/historical, not a persistent, directly-observable state.

**Why this manifests as a silently wrong number instead of an exception.**
This is a known industry phenomenon called **Silent Data Corruption (SDC)**:
GPU-fleet hardware faults that produce numerically wrong results without
ECC/ logging/exceptions firing, well documented at scale by Meta
("Silent Data Corruptions at Scale", 2020,
[research.facebook.com](https://research.facebook.com/publications/silent-data-corruptions-at-scale/);
also see [Meta's 2025 engineering-blog followup](https://engineering.fb.com/2025/07/22/data-infrastructure/how-meta-keeps-its-ai-hardware-reliable/))
and Google (SDC events reported weekly-to-biweekly during Gemini training,
[arxiv.org/html/2605.04213v1](https://arxiv.org/html/2605.04213v1)), with a
joint OCP whitepaper from Meta/Google/NVIDIA/Microsoft. NCCL and XLA's GPU
collectives trust the underlying transport and have no built-in
result-integrity verification (no equivalent of a network checksum); and
`donate_argnums`/`donate="all"` buffer donation is a static, compile-time
aliasing contract with no runtime liveness check (adding one would defeat
the point of zero-copy donation). If a degraded link delivers stale/wrong
bytes to a collective without violating the transport's own low-level
checks, or a donated buffer is reused for output before every collective
participant has finished, nothing in the JAX/XLA/NCCL stack detects it --
by design, not oversight, per the sources above.

**Consequence:** any `train()` call made with more than one JAX device
visible and no explicit device pinning, **on hardware with an undetected
latent PCIe/collective fault**, can train against a silently wrong loss/ELBO
without any error, crash, or log entry. On known-good hardware (e.g.
palladium, per current evidence) this specific wrong-answer mode was not
observed; the n=2 hang remains a risk regardless of host.

**Workaround (in use now, sufficient for both failure modes):** pin the
process to a single visible GPU via `CUDA_VISIBLE_DEVICES=<id>` before
importing `jax`. This collapses `len(jax.devices())` to 1, so the mesh is
trivial and no cross-device reduction/collective is needed -- correctness no
longer depends on either buggy/faulty code path. For an
embarrassingly-parallel multi-run campaign (e.g. many seeds/configs),
round-robin one GPU per process rather than sharding one process across
several GPUs. If multi-GPU sharding is required on magnesium specifically,
prefer restricting to NUMA0 (GPUs 0-3, zero wrong-answer instances observed)
and avoid exactly-2-GPU meshes on any host.

**Not yet done:**
- Physical inspection/reseat of magnesium's NUMA-node-1 GPUs (4-7) and their
  shared riser/PCIe switch -- the actual fix for the wrong-answer mode, vs.
  the software workaround above.
- File the n=2 hang as a jax-ml/jax upstream issue with the minimal repro
  (`fresh_process_sweep.sh`) -- this failure mode reproduced cleanly on both
  tested hosts and looks like a genuine, host-independent JAX/XLA collective
  deadlock, unlike the wrong-answer mode.
- Consider making `donate="all"` configurable in `trainer.py`'s
  `train_step` (e.g. a `donate: bool = True` knob) so multi-GPU sharded
  training can trade the memory-efficiency benefit of donation for
  correctness robustness on hardware of uncertain reliability, without code
  changes beyond a config flag.
- Consider a cheap periodic sanity check (recompute loss on a single
  unsharded device every N steps and compare within tolerance) as a
  defense-in-depth measure against undetected SDC, per Meta/OCP mitigation
  guidance -- would not prevent corruption but would catch it early instead
  of training silently on a wrong objective for many steps.
- Audit whether *any* prior multi-GPU `train()` run in this repo or
  downstream projects (any run launched without `CUDA_VISIBLE_DEVICES`
  pinning, especially on magnesium's NUMA1 GPUs) used a silently wrong loss.
  In the eeg downstream project specifically,
  `scripts/shell/hdsr_runs/current/run_jaxfads_hbn.sh` already pins
  `CUDA_VISIBLE_DEVICES` to a single GPU by default with an inline comment
  referencing this doc.
