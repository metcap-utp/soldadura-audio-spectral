"""Utilidades para medición de tiempos de ejecución."""

import time
from contextlib import contextmanager
from functools import wraps


class Timer:
    """Context manager y decorador para medir tiempos."""
    
    def __init__(self, name: str = None):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        if self.name:
            print(f"  [{self.name}] {self.elapsed:.3f}s")
    
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(self.name or func.__name__):
                return func(*args, **kwargs)
        return wrapper


@contextmanager
def timer(name: str):
    """Context manager simple para medir tiempo."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"  [{name}] {elapsed:.3f}s")
