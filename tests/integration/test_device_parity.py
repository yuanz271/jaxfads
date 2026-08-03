"""Opt-in CPU/GPU and single-/multi-GPU training integration tests."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_CASE = Path(__file__).with_name("run_device_case.py")


def _run_case(tmp_path: Path, *, platform: str, visible_devices: str | None):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = platform
    if visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = visible_devices
    output = tmp_path / platform.replace(",", "_")
    command = [sys.executable, str(_CASE), "--output", str(output)]
    try:
        subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    except FileNotFoundError as error:
        pytest.fail(f"could not launch device case: {error}")
    except subprocess.CalledProcessError as error:
        pytest.fail(
            f"device case failed ({platform}, {visible_devices}):\n"
            f"stdout:\n{error.stdout}\nstderr:\n{error.stderr}"
        )
    return output


def _assert_parity(reference: Path, candidate: Path):
    reference_meta = json.loads((reference / "metadata.json").read_text())
    candidate_meta = json.loads((candidate / "metadata.json").read_text())
    assert reference_meta["model_conf"] == candidate_meta["model_conf"]
    assert reference_meta["trainer_conf"] == candidate_meta["trainer_conf"]

    reference_arrays = np.load(reference / "result.npz")
    candidate_arrays = np.load(candidate / "result.npz")
    assert reference_arrays.files == candidate_arrays.files
    for name in reference_arrays.files:
        np.testing.assert_allclose(
            reference_arrays[name],
            candidate_arrays[name],
            rtol=2e-4,
            atol=2e-5,
            err_msg=name,
        )


def _device_count(platform: str):
    code = "import jax; print(len(jax.devices('gpu')))"
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = platform
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return 0
    return int(result.stdout.strip())


@pytest.mark.integration
def test_cpu_gpu_and_multi_gpu_training_parity(tmp_path):
    """A deterministic fit runs on each available device configuration."""
    if os.environ.get("JAXFADS_RUN_DEVICE_INTEGRATION") != "1":
        pytest.skip("set JAXFADS_RUN_DEVICE_INTEGRATION=1 to run device integration")

    cpu = _run_case(tmp_path, platform="cpu", visible_devices=None)
    gpu_count = _device_count("gpu")
    if gpu_count == 0:
        pytest.skip("no GPU backend is available")

    one_gpu = _run_case(tmp_path, platform="gpu", visible_devices="0")
    one_gpu_meta = json.loads((one_gpu / "metadata.json").read_text())
    assert one_gpu_meta["backend"] == "gpu"
    assert one_gpu_meta["device_count"] == 1
    _assert_parity(cpu, one_gpu)

    if gpu_count >= 2:
        two_gpu = _run_case(tmp_path, platform="gpu", visible_devices="0,1")
        two_gpu_meta = json.loads((two_gpu / "metadata.json").read_text())
        assert two_gpu_meta["backend"] == "gpu"
        assert two_gpu_meta["device_count"] == 2
        _assert_parity(cpu, two_gpu)

    cpu_meta = json.loads((cpu / "metadata.json").read_text())
    assert cpu_meta["backend"] == "cpu"
    assert cpu_meta["device_count"] == 1
