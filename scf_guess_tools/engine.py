from .metric import Metric
from .molecule import Molecule
from .wavefunction import Wavefunction
from abc import ABC, abstractmethod


class Engine(ABC):
    @abstractmethod
    def load(self, path: str) -> Molecule:
        pass

    @abstractmethod
    def guess(self, molecule: Molecule, basis: str, method: str) -> Wavefunction:
        pass

    @abstractmethod
    def calculate(
        self, molecule: Molecule, basis: str, guess: str | Wavefunction
    ) -> Wavefunction:
        pass

    @abstractmethod
    def score(
        self, initial: Wavefunction, final: Wavefunction, metric: Metric
    ) -> float:
        pass
