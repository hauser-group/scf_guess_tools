from __future__ import annotations

from .core import Object
from .builder import builder_property
from .matrix import Matrix
from .molecule import Molecule
from abc import ABC, abstractmethod

import numpy as np


class Wavefunction(Object, ABC):
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
        self._molecule = molecule
        self._basis = basis
        self._initial = initial
        self._origin = origin
        self._time = time
        self._iterations = iterations
        self._retried = retried
        self._converged = converged

    def __getstate__(self):
        return (
            Object.__getstate__(self),
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
        Object.__setstate__(self, serialized[0])

        (
            self._molecule,
            self._basis,
            self._initial,
            self._origin,
            self._time,
            self._iterations,
            self._retried,
            self._converged,
        ) = serialized[1:]
