"""Small logging helper so every module shares one configured logger."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "roofsolar") -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        root = logging.getLogger("roofsolar")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        root.propagate = False
        _CONFIGURED = True
    return logging.getLogger(name if name.startswith("roofsolar") else f"roofsolar.{name}")
