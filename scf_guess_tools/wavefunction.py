from __future__ import annotations

from .matrix import Matrix
from .molecule import Molecule
from abc import ABC, abstractmethod


class Wavefunction(ABC):
    def __init__(
        self,
        molecule: Molecule,
        basis: str,
        initial: str | Wavefunction = None,
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
    def initial(self) -> str | Wavefunction:
        return self._initial

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def retried(self) -> bool:
        return self._retried

    @property
    @abstractmethod
    def S(self) -> Matrix:
        pass

    @property
    @abstractmethod
    def D(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @property
    @abstractmethod
    def F(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @classmethod
    @abstractmethod
    def guess(cls, molecule: Molecule, basis: str, scheme: str) -> Wavefunction:
        pass

    @classmethod
    @abstractmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction
    ) -> Wavefunction:
        pass

    def __eq__(self, other: Wavefunction) -> bool:
        return (
            self.molecule == other.molecule
            and self.basis == other.basis
            and self.initial == other.initial
            and self.iterations == other.iterations
            and self.retried == other.retried
            and self.S == other.S
            and self.F == other.F
            and self.D == other.D
        )

    def __getstate__(self):
        return (self.molecule, self.basis, self.initial, self.iterations, self.retried)

    def __setstate__(self, serialized):
        (
            self._molecule,
            self._basis,
            self._initial,
            self._iterations,
            self._retried,
        ) = serialized
