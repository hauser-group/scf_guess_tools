from __future__ import annotations

from .common import Singleton
from .molecule import MoleculeBuilder
from .wavefunction import WavefunctionBuilder
from abc import ABC, abstractmethod
from joblib import Memory

import os


class Engine(MoleculeBuilder, WavefunctionBuilder, ABC, metaclass=Singleton):
    def __init__(self, cache: str, verbose: int):
        if cache is None:
            return

        directory = os.environ.get("SGT_CACHE")

        if directory is None:
            raise RuntimeError("SGT_CACHE environment variable not set")

        self._memory = Memory(directory, verbose=verbose)

        self.guess = self._memory.cache(self.guess, ignore=["self"])
        self.calculate = self._memory.cache(self.calculate, ignore=["self"])

    @property
    def memory(self) -> Memory:
        return self._memory

    @classmethod
    @abstractmethod
    def __repr__(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def guessing_schemes(cls) -> list[str]:
        pass
