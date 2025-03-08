from __future__ import annotations

from .common import Singleton
from .molecule import MoleculeBuilder
from .wavefunction import WavefunctionBuilder
from abc import ABC, abstractmethod
from joblib import Memory
from time import process_time

import functools
import os


def _double_time(inner, outer, *outer_args, **outer_kwargs):
    @functools.wraps(outer)
    def outer_wrapper(*args, **kwargs):
        @functools.wraps(inner)
        def inner_wrapper(*args, **kwargs):
            return inner(*args, **kwargs)

        start = process_time()
        result = outer(inner_wrapper, *outer_args, **outer_kwargs)(*args, **kwargs)
        result.load_time = process_time() - start

        return result

    return outer_wrapper


class Engine(MoleculeBuilder, WavefunctionBuilder, ABC, metaclass=Singleton):
    def __init__(
        self, cache: str, verbose: int = 0, cached_properties: list[str] | None = None
    ):
        self._cached_properties = cached_properties or []

        if cache:
            directory = os.environ.get("SGT_CACHE")

            if directory is None:
                raise RuntimeError("SGT_CACHE environment variable not set")

            self._memory = Memory(f"{directory}/{cache}", verbose=verbose)
            self.guess = _double_time(self.guess, self._memory.cache)
            self.calculate = _double_time(self.calculate, self._memory.cache)
        else:
            self._memory = None

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
