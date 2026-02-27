"""
Exponential-family variational distributions for XFADS.

Concrete ``Approx`` implementations live in submodules; this package
re-exports the public API.
"""

from .mvn import LoRaMVN, MVN

__all__ = ["MVN", "LoRaMVN"]
