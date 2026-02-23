"""
Dynamics models for XFADS.

This module implements concrete dynamics models for state transitions in
XFADS. Abstract interfaces are defined in ``jaxfads.base``.

.. deprecated::
    ``predict_moment`` and ``sample_expected_moment`` have moved to
    :mod:`jaxfads.core`.  Imports from this module still work but may
    be removed in a future release.
"""

from .core import predict_moment, sample_expected_moment  # noqa: F401 — backward compat
from .noise import DiagGaussian  # noqa: F401 — backward compat
