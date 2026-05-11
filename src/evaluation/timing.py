"""
src.evaluation.timing — wall-clock timing context manager.
"""

from __future__ import annotations

import time


class TimingContext:
    """Context manager that measures elapsed wall-clock time.

    Usage
    -----
    with TimingContext() as tc:
        model.fit(X_train, y_train)
    print(f"Elapsed: {tc.elapsed:.3f}s")
    """

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "TimingContext":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
