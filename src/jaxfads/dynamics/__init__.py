"""Dynamics models for XFADS.

Importing this package registers built-in subclasses via
`SubclassRegistryMixin.__init_subclass__`.

Built-ins
---------
- `OUDynamics`: zero-mean Ornstein–Uhlenbeck drift (diffusion-style tracking prior)
"""

from .ou import OUDynamics

__all__ = ["OUDynamics"]
