"""Shared logging configuration.

One logger per stage, all routed through the same formatter/handler so
pipeline runs produce a single readable log stream (and, optionally, a
log file per run for audit trails).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_CONFIGURED = False


def setup_logging(log_dir: str | Path | None = None, level: str = "INFO") -> None:
    """Configure root logging once per process.

    Args:
        log_dir: if given, also writes ``pipeline.log`` there.
        level: logging level name, e.g. "INFO", "DEBUG".
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "pipeline.log"))

    fmt = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"
    logging.basicConfig(level=level.upper(), format=fmt, handlers=handlers, force=True)
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, configuring logging on first use."""
    if not _CONFIGURED:
        setup_logging()
    return logging.getLogger(name)
