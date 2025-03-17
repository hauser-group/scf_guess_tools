from abc import ABC, abstractmethod
from enum import Enum
from functools import wraps
from joblib import Memory

import os


class Backend(Enum):
    PSI = "Psi"
    PY = "Py"


class Object(ABC):
    def __getstate__(self):
        return self.backend()

    def __setstate__(self, serialized):
        assert serialized == self.backend()

    @classmethod
    @abstractmethod
    def backend(self) -> Backend:
        pass


_memory = None
_cache_verbosity = None


def cache_directory(throw: bool = False) -> str | None:
    directory = os.environ.get("SGT_CACHE")

    if throw and directory is None:
        raise RuntimeError("SGT_CACHE environment variable is not set")

    return directory


def cache(ignore: list[str] = None, verbose: int = None):
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
    _cache_verbosity = level


def clear_cache():
    try:
        global _memory
        _memory.clear()
    except:
        pass


def reset():
    global _memory
    global _cache_verbosity

    clear_cache()
    _memory = None
    _cache_verbosity = None
