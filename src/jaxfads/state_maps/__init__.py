"""State-map models for XFADS.

Importing this package registers built-in subclasses via
`SubclassRegistryMixin.__init_subclass__`.

Built-ins
---------
- `OUStateMap`: zero-mean Ornstein–Uhlenbeck drift map
- `FunctionStateMap`: wrapper around plain Python functions/methods/partials
"""

from .function import FunctionStateMap
from .ou import OUStateMap

__all__ = ["OUStateMap", "FunctionStateMap"]
