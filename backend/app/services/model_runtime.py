"""Shared safeguards for memory-constrained ML inference deployments."""
from __future__ import annotations

from functools import wraps
import ctypes
import gc
import os
from pathlib import Path
import threading


MODEL_OPERATION_LOCK = threading.RLock()


def serialized_model_operation(function):
    """Run only one model-backed request at a time in this process."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with MODEL_OPERATION_LOCK:
            return function(*args, **kwargs)

    return wrapped


def collect_released_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        pass
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def current_rss_mb() -> float | None:
    """Read current resident memory without adding a monitoring dependency."""
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
        return round(resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024 / 1024, 1)
    except (OSError, ValueError, IndexError):
        return None
