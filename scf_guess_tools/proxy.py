from __future__ import annotations

from .core import Backend
from typing import Callable


def proxy(backend: Backend, resolve: Callable, *args, **kwargs):
    """Execute a function defined for a specified backend.

    Args:
        backend: The backend to use.
        resolve: A callable returning the function to be invoked.
        *args: Positional arguments passed to the function.
        **kwargs: Keyword arguments passed to the function.

    Returns:
        The function result.
    """
    package = None

    if backend == Backend.PSI:
        from . import psi

        package = psi
    elif backend == Backend.PY:
        from . import py

        package = py
    else:
        raise ValueError(f"Invalid backend {backend}")

    operation = resolve(package)
    return operation(*args, **kwargs)
