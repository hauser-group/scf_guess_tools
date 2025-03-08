from __future__ import annotations

from .common import Singleton, double_time, single_time
from .molecule import MoleculeBuilder
from .wavefunction import WavefunctionBuilder
from abc import ABC, abstractmethod
from joblib import Memory

import os


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
            self.guess = self._memory.cache(self.guess)
            self.calculate = self._memory.cache(self.calculate)
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
