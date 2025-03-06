from __future__ import annotations

from .engine import Engine
from .matrix import Matrix
from .molecule import Molecule
from abc import ABC, abstractmethod


class Wavefunction(ABC):
    @property
    def engine(self) -> Engine:
        return self._engine

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
    def converged(self) -> bool:
        return self._converged

    @property
    @abstractmethod
    def S(self) -> Matrix:
        pass

    @property
    @abstractmethod
    def H(self) -> Matrix:
        pass

    @property
    @abstractmethod
    def D(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @property
    @abstractmethod
    def F(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @property
    def energy(self) -> float:  # TODO check for memory inefficiencies
        if self.molecule.singlet:
            # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3a_restricted-hartree-fock.ipynb

            result = self.F + self.H
            result = result @ self.D

            return result.trace
        else:
            # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3c_unrestricted-hartree-fock.ipynb

            Da, Db = self.D
            Fa, Fb = self.F

            terms = [(a @ b).trace for a, b in zip([Da + Db, Da, Db], [self.H, Fa, Fb])]
            return 0.5 * sum(terms)

    def __init__(
        self,
        engine: Engine,
        molecule: Molecule,
        basis: str,
        initial: str | Wavefunction = None,
        iterations: int = None,
        retried: bool = None,
        converged: bool = None,
    ):
        self._engine = engine
        self._molecule = molecule
        self._basis = basis
        self._initial = initial
        self._iterations = iterations
        self._retried = retried
        self._converged = converged

    def __eq__(self, other: Wavefunction) -> bool:
        return (
            self.molecule == other.molecule
            and self.basis == other.basis
            and self.initial == other.initial
            and self.iterations == other.iterations
            and self.retried == other.retried
            and self.converged == other.converged
            and self.S == other.S
            and self.D == other.D
            and self.F == other.F
            and self.H == other.H
            and self.energy == other.energy
        )

    def __getstate__(self):
        return (
            self.molecule,
            self.basis,
            self.initial,
            self.iterations,
            self.retried,
            self.converged,
        )

    def __setstate__(self, serialized):
        (
            self._molecule,
            self._basis,
            self._initial,
            self._iterations,
            self._retried,
            self._converged,
        ) = serialized

    @classmethod
    @abstractmethod
    def guess(
        cls, engine: Engine, molecule: Molecule, basis: str, scheme: str
    ) -> Wavefunction:
        pass

    @classmethod
    @abstractmethod
    def calculate(
        cls,
        engine: Engine,
        molecule: Molecule,
        basis: str,
        guess: str | Wavefunction | None = None,
    ) -> Wavefunction:
        pass
