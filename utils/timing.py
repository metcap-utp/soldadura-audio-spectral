"""Utilidades para medición de tiempos de ejecución — Compatible con vggish-backbone y yamnet-backbone."""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator, Optional


@dataclass(frozen=True)
class TimingResult:
    """Resultado de medición de tiempo con propiedades de conversión."""
    seconds: float

    @property
    def minutes(self) -> float:
        return self.seconds / 60

    @property
    def hours(self) -> float:
        return self.seconds / 3600


@contextmanager
def timer(
    label: str,
    *,
    print_fn: Callable[[str], None] = print,
    enabled: bool = True,
) -> Iterator[Callable[[], TimingResult]]:
    """Simple context-manager timer compatible with vggish/yamnet.

    Usage:
        with timer("Extracción features") as get_elapsed:
            ...
        result = get_elapsed()
        extraction_time = result.seconds
    """

    if not enabled:

        def _get_elapsed_disabled() -> TimingResult:
            return TimingResult(0.0)

        yield _get_elapsed_disabled
        return

    start = time.perf_counter()

    def _get_elapsed() -> TimingResult:
        return TimingResult(time.perf_counter() - start)

    print_fn(f"[TIMER] {label}...")
    try:
        yield _get_elapsed
    finally:
        elapsed = _get_elapsed().seconds
        print_fn(f"[TIMER] {label}: {elapsed:.2f}s ({elapsed / 60:.2f}min)")


class Timer:
    """Backward-compatible Timer class for code that uses `with Timer("name")`."""
    
    def __init__(self, name: str = None):
        self.name = name
        self.start_time = None
        self.elapsed = None
        self._ctx = None
        self._get_elapsed = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self._ctx = timer(self.name or "Timer", print_fn=print if self.name else lambda x: None, enabled=True)
        self._get_elapsed = self._ctx.__enter__()
        return self
    
    def __exit__(self, *args):
        self._ctx.__exit__(*args)
        if self._get_elapsed:
            self.elapsed = self._get_elapsed().seconds
        else:
            self.elapsed = time.perf_counter() - self.start_time
