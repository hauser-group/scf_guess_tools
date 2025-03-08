from __future__ import annotations

from .common import Singleton
from .molecule import MoleculeBuilder
from .wavefunction import WavefunctionBuilder
from abc import ABC, abstractmethod
from joblib import Memory
from time import process_time

import functools
import os


def _nested_time(inner, outer, *outer_args, **outer_kwargs):
    if getattr(inner, "__self__", None) is not None:
        inner = inner.__func__

    @functools.wraps(outer)
    def outer_wrapper(*args, **kwargs):
        start = process_time()
        result = outer(inner, *outer_args, **outer_kwargs)(*args, **kwargs)
        result.load_time = process_time() - start
        return result

    setattr(outer_wrapper, "_double_time_inner", inner)
    return outer_wrapper


class Engine(MoleculeBuilder, WavefunctionBuilder, ABC, metaclass=Singleton):
    def __init__(
        self, cache: str, verbose: int = 0, cached_properties: list[str] | None = None
    ):
        self._cached_properties = cached_properties or []

        if cache and getattr(self, "_memory", None) is None:
            directory = os.environ.get("SGT_CACHE")

            if directory is None:
                raise RuntimeError("SGT_CACHE environment variable not set")

            self._memory = Memory(f"{directory}/{cache}", verbose=verbose)

        for method in ["guess", "calculate"]:
            cls = self.__class__
            function = getattr(cls, method)
            function = getattr(function, "_double_time_inner", function)

            if getattr(function, "__self__", None) is not None:
                function = function.__func__

            if cache:
                function = _nested_time(function, self._memory.cache)

            setattr(cls, method, function)

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def cached_properties(self) -> list[str]:
        return self._cached_properties

    @classmethod
    @abstractmethod
    def __repr__(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def guessing_schemes(cls) -> list[str]:
        pass
