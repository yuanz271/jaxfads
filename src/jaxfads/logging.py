"""
Logging utilities for jaxfads.

This module is deliberately host-side only. Do not call Python logging APIs from
functions that may be JIT-compiled; for deep debugging of JAX-transformed code,
collect scalars via helper functions and log them outside of JIT.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any


_BASE_LOGGER_NAME = "jaxfads"


def get_logger(name: str = _BASE_LOGGER_NAME) -> logging.Logger:
    """
    Return a namespaced jaxfads logger.

    Parameters
    ----------
    name : str, default=_BASE_LOGGER_NAME
        Logger name. If ``name`` is a module name under ``jaxfads`` (e.g.
        ``"jaxfads.trainer"``), it is used directly. Otherwise it is prefixed with
        ``"jaxfads."``.

    Returns
    -------
    logging.Logger
        Logger instance scoped under the jaxfads namespace.
    """
    if name == _BASE_LOGGER_NAME or name.startswith(_BASE_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_BASE_LOGGER_NAME}.{name}")


def _find_handler(
    logger: logging.Logger, handler_type: type[logging.Handler], **attrs: Any
) -> logging.Handler | None:
    """
    Return the first handler whose type matches *exactly* (not via subclass).

    Parameters
    ----------
    logger : logging.Logger
        Logger to search.
    handler_type : type[logging.Handler]
        Handler class to match exactly.
    **attrs : Any
        Attribute filters that must match on the handler instance.

    Returns
    -------
    logging.Handler | None
        Matching handler or ``None`` if no handler is found.
    """
    for h in logger.handlers:
        if type(h) is not handler_type:
            continue
        if all(getattr(h, k, None) == v for k, v in attrs.items()):
            return h
    return None


def configure_logging(
    level: str | int = "INFO",
    *,
    file_path: str | os.PathLike[str] | None = None,
) -> None:
    """
    Configure jaxfads logging.

    This function is idempotent: calling it multiple times will not add
    duplicate handlers. Calling it again with a different *level* updates
    the logger and all previously-attached handlers.

    Console output always uses Rich's ``RichHandler``.

    Parameters
    ----------
    level : str | int, default="INFO"
        Logging level for the base ``jaxfads`` logger. Can be overridden by
        the ``JAXFADS_LOG_LEVEL`` environment variable (env var always wins).
    file_path : str | os.PathLike[str] | None, default=None
        Optional log file path. If not provided, reads ``JAXFADS_LOG_FILE``.
    """
    env_level = os.environ.get("JAXFADS_LOG_LEVEL")
    if env_level:
        level = env_level.upper()

    logger = logging.getLogger(_BASE_LOGGER_NAME)
    logger.setLevel(level)

    # Ensure child loggers propagate to this base logger.
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # --- console handler (always Rich) ---
    from rich.logging import RichHandler

    existing = _find_handler(logger, RichHandler)
    if existing is not None:
        existing.setLevel(level)
    else:
        rh = RichHandler(
            rich_tracebacks=True,
            markup=False,
            show_time=False,
            show_level=False,
            show_path=False,
        )
        rh.setLevel(level)
        rh.setFormatter(formatter)
        logger.addHandler(rh)

    # --- file handler ---
    if file_path is None:
        env_file = os.environ.get("JAXFADS_LOG_FILE")
        file_path = env_file if env_file else None

    if file_path is not None:
        path = Path(file_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(path.resolve())
        existing = _find_handler(logger, logging.FileHandler, baseFilename=resolved)
        if existing is not None:
            existing.setLevel(level)
        else:
            for handler in list(logger.handlers):
                if isinstance(handler, logging.FileHandler):
                    handler.close()
                    logger.removeHandler(handler)
            fh = logging.FileHandler(resolved, encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
