"""Contracts for trainer-owned model transformations."""

from __future__ import annotations

from typing import Any, Protocol


class ModelTransformation(Protocol):
    """One ordered, trainer-owned model transformation."""

    def initialize(self, model, *, key) -> Any: ...

    def __call__(self, model, batch, forward, *, key) -> Any: ...

    def frozen_paths(self, model) -> list[str]: ...
