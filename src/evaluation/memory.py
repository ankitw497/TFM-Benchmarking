"""
src.evaluation.memory — peak memory tracking.

Delegates to psutil when available; returns 0.0 otherwise so benchmarks
run cleanly on machines without psutil installed.
"""

from __future__ import annotations


def get_peak_memory() -> float:
    """Return current RSS memory usage of this process in MB.

    Returns
    -------
    float
        Memory in megabytes, or 0.0 if psutil is not installed.
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0
