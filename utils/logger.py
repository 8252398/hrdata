# -*- coding: utf-8 -*-
"""Unified logging — no print() in production code."""

import logging
import sys
from pathlib import Path

LOG_FILE = Path(__file__).resolve().parent.parent / "app_hr.log"

_logger: logging.Logger | None = None


def get_logger(name: str = "app_hr") -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger

    _logger = logging.getLogger(name)
    _logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    _logger.addHandler(fh)

    # Console handler (warnings only, to avoid cluttering terminal)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt)
    _logger.addHandler(ch)

    return _logger
