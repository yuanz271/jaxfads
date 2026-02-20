"""
Proof: checkpoint placement matters for lax.scan memory.

Compares three gradient-through-scan strategies:
  1. plain    — no checkpoint at all
  2. outer    — checkpoint wraps the function containing scan (current jaxfads)
  3. body     — checkpoint wraps the scan step function (proposed)

Runs each in a subprocess for isolated peak-RSS measurement.

Usage:  uv run python tests/bench_checkpoint.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap


def _make_code(D: int, T: int, mode: str) -> str:
    return textwrap.dedent(f"""\
    import gc, json, os, platform, resource, time
    os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

    import jax
    import jax.numpy as jnp
    from jax import checkpoint, grad, lax, random

    D, T, MODE = {D}, {T}, "{mode}"

    keys = random.split(random.key(0), 4)
    W1 = random.normal(keys[0], (D, D)) * D**-0.5
    W2 = random.normal(keys[1], (D, D)) * D**-0.5
    W3 = random.normal(keys[2], (D, D)) * D**-0.5
    init = jnp.zeros(D)
    xs   = random.normal(keys[3], (T, D)) * 0.01

    def body(carry, x):
        h = jnp.tanh(carry @ W1 + x)
        h = jnp.tanh(h @ W2)
        h = jnp.tanh(h @ W3)
        return h, h

    def f_plain(init, xs):
        _, ys = lax.scan(body, init, xs)
        return jnp.sum(ys)

    def f_outer(init, xs):
        return checkpoint(f_plain)(init, xs)

    def f_body(init, xs):
        _, ys = lax.scan(checkpoint(body), init, xs)
        return jnp.sum(ys)

    fn = dict(plain=f_plain, outer=f_outer, body=f_body)[MODE]

    g = jax.jit(grad(fn))

    # Warm up
    _ = g(init, xs).block_until_ready()
    gc.collect()

    # Timed run
    t0 = time.perf_counter()
    grads = g(init, xs)
    grads.block_until_ready()
    dt = time.perf_counter() - t0

    raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = raw_rss / (1024 * 1024) if platform.system() == "Darwin" else raw_rss / 1024

    print(json.dumps(dict(
        mode=MODE, D=D, T=T,
        peak_rss_mb=round(peak_mb, 1),
        time_s=round(dt, 4),
    )))
    """)


def _run(D: int, T: int, mode: str, timeout: int = 180) -> dict:
    code = _make_code(D, T, mode)
    env = {**os.environ, "JAX_PLATFORM_NAME": "cpu"}
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"mode": mode, "D": D, "T": T, "error": "TIMEOUT"}
    if proc.returncode != 0:
        err = (proc.stderr or "")[-300:].strip()
        return {"mode": mode, "D": D, "T": T, "error": err}
    for line in reversed(proc.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"mode": mode, "D": D, "T": T, "error": "no JSON output"}


def main() -> None:
    configs = [
        (128,  2_000),
        (128, 10_000),
        (128, 40_000),
        (256,  2_000),
        (256, 10_000),
    ]
    modes = ["plain", "outer", "body"]

    print(f"{'D':>5}  {'T':>7}  {'mode':<7}  {'peak RSS MB':>12}  {'time (s)':>9}")
    print("-" * 48)

    for D, T in configs:
        for mode in modes:
            r = _run(D, T, mode)
            if "error" in r:
                print(f"{D:>5}  {T:>7}  {mode:<7}  {'ERR':>12}  {r['error'][:50]}")
            else:
                print(
                    f"{D:>5}  {T:>7}  {mode:<7}"
                    f"  {r['peak_rss_mb']:>12.1f}"
                    f"  {r['time_s']:>9.4f}"
                )
        print()


if __name__ == "__main__":
    main()
