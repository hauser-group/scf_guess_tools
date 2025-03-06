from __future__ import annotations

from abc import ABC, abstractmethod
from joblib import Memory

import os
import scf_guess_tools.molecule as m
import scf_guess_tools.wavefunction as w


class Engine(ABC):
    def __init__(self, cache: str, verbose: int):
        if cache is None:
            return

        base = os.environ.get("SGT_CACHE")

        if base is None:
            raise RuntimeError("SGT_CACHE environment variable not set")

        self._memory = Memory(base, verbose=verbose)

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

    @abstractmethod
    def load(self, path: str) -> m.Molecule:
        pass

    @abstractmethod
    def guess(self, molecule: m.Molecule, basis: str, scheme: str) -> w.Wavefunction:
        pass

    @abstractmethod
    def calculate(
        self,
        molecule: m.Molecule,
        basis: str,
        guess: str | w.Wavefunction | None = None,
    ) -> w.Wavefunction:
        pass
