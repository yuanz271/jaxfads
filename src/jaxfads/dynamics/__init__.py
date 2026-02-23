"""
Dynamics models for XFADS.

This subpackage provides concrete dynamics implementations and
backward-compatible re-exports. Abstract interfaces are defined
in :mod:`jaxfads.base`.
"""

from ..core import predict_moment, sample_expected_moment  # noqa: F401
from .noise import DiagGaussian  # noqa: F401
