from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from joblib import Memory

import os


class Backend(Enum):
    """Available computational backends."""

    PSI = "Psi"
    PY = "Py"


class Object(ABC):
    """Base class for objects that depend on a backend."""

    def __getstate__(self):
        """Return a serialized state for pickling.

        Returns:
            The backend associated with this object.
        """
        return self.backend()

    def __setstate__(self, serialized):
        """Restore the object from a serialized state.

        Args:
            serialized: The serialized backend state.
        """
        assert serialized == self.backend()

    @classmethod
    @abstractmethod
    def backend(self) -> Backend:
        """Return the backend associated with this object."""
        pass


_memory = None
_cache_verbosity = None


def guessing_schemes(backend: Backend) -> list[str]:
    """Return valid initial guessing schemes for a backend.

    Args:
        backend: The backend for which guessing schemes are retrieved.

    Returns:
        A list of valid initial guess strings for the given backend.
    """
    if backend == Backend.PSI:
        from . import psi

        return psi.guessing_schemes
    elif backend == Backend.PY:
        from . import py

        return py.guessing_schemes
    else:
        raise ValueError(f"Unknown backend: {backend}")


def cache_directory(throw: bool = False) -> str | None:
    """Get the base cache directory from the SGT_CACHE environment variable.

    Args:
        throw: If True, raise an exception if the cache directory is not set.

    Returns:
        The cache directory path, or None if not set.
    """
    directory = os.environ.get("SGT_CACHE")

    if throw and directory is None:
        raise RuntimeError("SGT_CACHE environment variable is not set")

    return directory


def cache(ignore: list[str] = None, verbose: int = None):
    """Enable caching for a function using joblib.Memory. Functions decorated with this
    gain a cache flag (default: True). If it is enabled, function results are cached on
    disk.

    Args:
        ignore: A list of argument names to exclude from the key hash.
        verbose: Verbosity level for joblib.

    Returns:
        A decorator that enables caching for the function.
    """

    def decorator(function):
        @wraps(function)
        def wrapper(*args, cache: bool = True, **kwargs):
            if not cache:
                return function(*args, **kwargs)

            global _memory

            if _memory is None:
                directory = cache_directory(throw=True)
                _memory = Memory(directory)

            v = _cache_verbosity if verbose is None else verbose
            return _memory.cache(function, ignore=ignore, verbose=v)(*args, **kwargs)

        return wrapper

    return decorator


def cache_verbosity(level: int):
    """Set the cache verbosity level globally.

    Args:
        level: The verbosity level for caching.
    """
    global _cache_verbosity
    _cache_verbosity = level


def clear_cache():
    """Clear the cached function results."""
    try:
        global _memory
        _memory.clear()
    except:
        pass


def reset():
    """Reset the package state and that of all subpackages."""
    global _memory
    global _cache_verbosity

    clear_cache()
    _memory = None
    _cache_verbosity = None
