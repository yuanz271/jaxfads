"""State-map models for XFADS.

Importing this package registers built-in subclasses via
`SubclassRegistryMixin.__init_subclass__`.

Built-ins
---------
- `OUStateMap`: zero-mean Ornstein–Uhlenbeck drift map
"""

from .ou import OUStateMap

__all__ = ["OUStateMap"]
