"""Functional non-trainable dynamics."""

from __future__ import annotations

import functools
import importlib
import inspect
import types
from collections.abc import Callable
from typing import Any

from jax import Array

from ..base import Dynamics


def _is_plain_function_or_method(fn: Any) -> bool:
    if isinstance(fn, functools.partial):
        return _is_plain_function_or_method(fn.func)
    return (
        isinstance(
            fn,
            (types.FunctionType, types.BuiltinFunctionType, types.MethodType),
        )
        or inspect.isfunction(fn)
        or inspect.ismethod(fn)
        or inspect.isbuiltin(fn)
    )


def _accepts_key_kwarg(fn: Callable[..., Any]) -> bool:
    sig = inspect.signature(fn)
    for p in sig.parameters.values():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if p.name == "key":
            return True
    return False


def _resolve_from_path(fn_path: str) -> Callable[..., Array]:
    if ":" not in fn_path:
        raise ValueError(
            "Functional expects dyn_conf.fn_path in 'module:function' format."
        )
    module_name, symbol_name = fn_path.split(":", 1)
    if not module_name or not symbol_name:
        raise ValueError(
            "Functional expects dyn_conf.fn_path in 'module:function' format."
        )
    module = importlib.import_module(module_name)
    try:
        fn = getattr(module, symbol_name)
    except AttributeError as e:
        raise ValueError(
            f"Functional could not find symbol '{symbol_name}' in module '{module_name}'."
        ) from e
    return fn


class Functional(Dynamics):
    """Wrap a plain Python function/method/partial as a non-trainable map."""

    fn: Callable[..., Array]
    takes_key: bool

    def __init__(self, conf, key: Array):
        self.conf = conf
        fn_path = getattr(conf, "fn_path", None)
        if fn_path is None:
            raise ValueError(
                "Functional requires `dyn_conf.fn_path` "
                "(format: 'module:function')."
            )
        fn = _resolve_from_path(str(fn_path))
        fn_kwargs = getattr(conf, "fn_kwargs", None)
        if fn_kwargs is not None:
            fn = functools.partial(fn, **dict(fn_kwargs))
        if not _is_plain_function_or_method(fn):
            raise TypeError(
                "Functional expects a plain Python function, method, "
                "or functools.partial of one."
            )
        self.fn = fn
        self.takes_key = _accepts_key_kwarg(fn)

    def eval(self, z: Array, u: Array, c: Array, *, key=None) -> Array:
        if self.takes_key:
            return self.fn(z, u, c, key=key)
        return self.fn(z, u, c)


__all__ = ["Functional"]
