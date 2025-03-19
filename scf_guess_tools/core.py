from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from joblib import Memory

import os
import shutil


class Backend(Enum):
    """Available computational backends."""

    PSI = "Psi"
    PY = "Py"


class Object(ABC):
    """Base class for objects that depend on a backend."""

    @abstractmethod
    def __hash__(self) -> int:
        """Return a deterministic hash that is the same for all object with the same
        properties according to the semantics of this package. This does not necessarily
        mean the underlying native objects are identical too!

        Returns:
            A hash value uniquely identifying the object.
        """
        pass

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


def clear_cache():
    """Clear the cache directory."""
    try:
        shutil.rmtree(cache_directory())
    except:
        pass


def reset():
    """Reset the package state and that of all subpackages."""
    pass
