from .molecule import Molecule
from abc import ABC, abstractmethod
from typing import Self


class Wavefunction(ABC):
    def __init__(
        self,
        molecule: Molecule,
        basis: str,
        initial: str | Self = None,
        iterations: int = None,
        retried: bool = None,
    ):
        self._molecule = molecule
        self._basis = basis
        self._initial = initial
        self._iterations = iterations
        self._retried = retried

    @property
    @abstractmethod
    def native(self):
        pass

    @property
    def molecule(self) -> Molecule:
        return self._molecule

    @property
    def basis(self) -> str:
        return self._basis

    @property
    def initial(self) -> str | Self:
        return self._initial

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def retried(self) -> bool:
        return self._retried

    @property
    @abstractmethod
    def S(self):
        pass

    @property
    @abstractmethod
    def D(self):
        pass

    @property
    @abstractmethod
    def F(self):
        pass

    @classmethod
    @abstractmethod
    def guess(cls, molecule: Molecule, basis: str, scheme: str) -> Self:
        pass

    @classmethod
    @abstractmethod
    def calculate(cls, molecule: Molecule, basis: str, guess: str | Self) -> Self:
        pass

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, serialized):
        pass
