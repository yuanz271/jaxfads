import logging
import os
from pathlib import Path

from rich.logging import RichHandler

from jaxfads.logging import configure_logging, get_logger


def _snapshot_base_logger():
    base = get_logger("jaxfads")
    return base, list(base.handlers), base.level, base.propagate


def _restore_base_logger(base, handlers, level, propagate):
    base.handlers = handlers
    base.setLevel(level)
    base.propagate = propagate


def test_configure_logging_idempotent(tmp_path: Path):
    base, handlers, level, propagate = _snapshot_base_logger()
    try:
        base.handlers = []
        configure_logging("INFO")
        n1 = len(base.handlers)
        configure_logging("INFO")
        n2 = len(base.handlers)
        assert n1 == n2
        assert n1 >= 1
    finally:
        _restore_base_logger(base, handlers, level, propagate)


def test_configure_logging_file_handler(tmp_path: Path):
    base, handlers, level, propagate = _snapshot_base_logger()
    old_env = os.environ.get("JAXFADS_LOG_FILE")
    try:
        base.handlers = []
        log_path = tmp_path / "jaxfads.log"
        os.environ["JAXFADS_LOG_FILE"] = str(log_path)
        configure_logging("INFO")

        log = get_logger("tests")
        log.info("file handler works")

        for h in base.handlers:
            try:
                h.flush()
            except Exception:
                pass

        text = log_path.read_text(encoding="utf-8")
        assert "file handler works" in text
    finally:
        if old_env is None:
            os.environ.pop("JAXFADS_LOG_FILE", None)
        else:
            os.environ["JAXFADS_LOG_FILE"] = old_env
        _restore_base_logger(base, handlers, level, propagate)


def test_configure_logging_updates_handler_level():
    """Calling configure_logging again with a new level updates existing handlers."""
    base, handlers, level, propagate = _snapshot_base_logger()
    try:
        base.handlers = []
        configure_logging("WARNING")
        assert len(base.handlers) == 1
        assert base.handlers[0].level == logging.WARNING

        configure_logging("DEBUG")
        # Handler count should not increase.
        assert len(base.handlers) == 1
        assert base.handlers[0].level == logging.DEBUG
        assert base.level == logging.DEBUG
    finally:
        _restore_base_logger(base, handlers, level, propagate)


def test_configure_logging_uses_rich_handler():
    """Console handler is always a RichHandler."""
    base, handlers, level, propagate = _snapshot_base_logger()
    try:
        base.handlers = []
        configure_logging("INFO")
        assert len(base.handlers) == 1
        assert isinstance(base.handlers[0], RichHandler)
    finally:
        _restore_base_logger(base, handlers, level, propagate)
