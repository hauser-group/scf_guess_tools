from __future__ import annotations

from .builder import Builder, builder_property
from .matrix import Matrix
from .molecule import Molecule
from abc import ABC, abstractmethod

import numpy as np


class WavefunctionBuilder(ABC):
    @classmethod
    @abstractmethod
    def guess(
        cls, molecule: Molecule, basis: str, scheme: str | None = None
    ) -> Wavefunction:
        pass

    @classmethod
    @abstractmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction | None = None
    ) -> Wavefunction:
        pass


class Wavefunction(Builder, WavefunctionBuilder, ABC):
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
    def origin(self) -> str:
        return self._origin

    @property
    def time(self) -> float:
        return self._time

    @property
    def load_time(self) -> float | None:
        return self._load_time

    @load_time.setter
    def load_time(self, load_time: float):
        self._load_time = load_time

    @property
    def iterations(self) -> int | None:
        return self._iterations

    @property
    def retried(self) -> bool | None:
        return self._retried

    @property
    def converged(self) -> bool | None:
        return self._converged

    @builder_property
    @abstractmethod
    def S(self) -> Matrix:
        pass

    @builder_property
    @abstractmethod
    def H(self) -> Matrix:
        pass

    @builder_property
    @abstractmethod
    def D(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @builder_property
    @abstractmethod
    def F(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @builder_property
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
        molecule: Molecule,
        basis: str,
        initial: str | Wavefunction,
        origin: str,
        time: float,
        iterations: int | None = None,
        retried: bool | None = None,
        converged: bool | None = None,
    ):
        Builder.__init__(self)
        self._load_time = None

        self._molecule = molecule
        self._basis = basis
        self._initial = initial
        self._origin = origin
        self._time = time
        self._iterations = iterations
        self._retried = retried
        self._converged = converged

    def __eq__(self, other: Wavefunction) -> bool:
        return (
            self.molecule == other.molecule
            and self.basis == other.basis
            and self.initial == other.initial
            and self.origin == other.origin
            and self.time == other.time
            and self.iterations == other.iterations
            and self.retried == other.retried
            and self.converged == other.converged
            and self.S == other.S
            and self.D == other.D
            and self.F == other.F
            and self.H == other.H
            and np.isclose(self.energy, other.energy, rtol=1e-5, atol=1e-10)
        )

    def __getstate__(self):
        return (
            self.molecule,
            self.basis,
            self.initial,
            self.origin,
            self.time,
            self.iterations,
            self.retried,
            self.converged,
        )

    def __setstate__(self, serialized):
        Builder.__init__(self)

        (
            self._molecule,
            self._basis,
            self._initial,
            self._origin,
            self._time,
            self._iterations,
            self._retried,
            self._converged,
        ) = serialized
