from .metric import Metric
from .molecule import Molecule
from .wavefunction import Wavefunction
from abc import ABC, abstractmethod
from joblib import Memory
import os


class Engine(ABC):
    def __init__(self, cache: str):
        if cache is None:
            return

        base = os.environ.get("SGT_CACHE")

        if base is None:
            raise RuntimeError("SGT_CACHE environment variable not set")

        self._memory = Memory(base, verbose=0)

        self.guess = self._memory.cache(self.guess, ignore=["self"])
        self.calculate = self._memory.cache(self.calculate, ignore=["self"])
        self.score = self._memory.cache(self.score, ignore=["self"])

    @property
    def memory(self) -> Memory:
        return self._memory

    @abstractmethod
    def load(self, path: str) -> Molecule:
        pass

    @abstractmethod
    def guess(self, molecule: Molecule, basis: str, method: str) -> Wavefunction:
        pass

    @abstractmethod
    def calculate(
        self, molecule: Molecule, basis: str, guess: str | Wavefunction = None
    ) -> Wavefunction:
        pass

    @abstractmethod
    def score(
        self, initial: Wavefunction, final: Wavefunction, metric: Metric
    ) -> float:
        pass

    @property
    @abstractmethod
    def guessing_schemes(self) -> list[str]:
        pass
