from __future__ import annotations

from .common import timeable, tuplifyable
from .core import Object
from .matrix import Matrix
from .molecule import Molecule
from abc import ABC, abstractmethod


class Wavefunction(Object, ABC):
    @property
    @abstractmethod
    def native(self):
        pass

    @property
    @abstractmethod
    def molecule(self) -> Molecule:
        pass

    @property
    def basis(self) -> str:
        return self._basis

    @property
    @abstractmethod
    def initial(self) -> str | Wavefunction:
        pass

    @property
    def origin(self) -> str:
        return self._origin

    @property
    def time(self) -> float:
        return self._time

    @property
    def converged(self) -> bool | None:
        return self._converged

    @property
    def stable(self) -> bool | None:
        return self._stable

    @property
    def second_order(self) -> bool | None:
        return self._second_order

    @abstractmethod
    @timeable
    def overlap(self) -> Matrix:
        pass

    @abstractmethod
    @timeable
    def core_hamiltonian(self) -> Matrix:
        pass

    @abstractmethod
    @timeable
    @tuplifyable
    def density(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @abstractmethod
    @timeable
    @tuplifyable
    def fock(self) -> Matrix | tuple[Matrix, Matrix]:
        pass

    @tuplifyable
    def electronic_energy(
        self,
        core_hamiltonian: Matrix | None = None,
        density: Matrix | tuple[Matrix, Matrix] = None,
        fock: Matrix | tuple[Matrix, Matrix] | None = None,
    ) -> float:  # TODO check for memory inefficiencies
        H = self.core_hamiltonian() if core_hamiltonian is None else core_hamiltonian
        D = self.density() if density is None else density
        F = self.fock() if fock is None else fock

        if self.molecule.singlet:
            # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3a_restricted-hartree-fock.ipynb

            result = F + H
            result = result @ D

            return result.trace
        else:
            # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3c_unrestricted-hartree-fock.ipynb

            Da, Db = D
            Fa, Fb = F

            terms = [(a @ b).trace for a, b in zip([Da + Db, Da, Db], [H, Fa, Fb])]
            return 0.5 * sum(terms)

    def __init__(
        self,
        basis: str,
        origin: str,
        time: float,
        converged: bool | None = None,
        stable: bool | None = None,
        second_order: bool | None = None,
    ):
        self._basis = basis
        self._origin = origin
        self._time = time
        self._converged = converged
        self._stable = stable
        self._second_order = second_order

    def __getstate__(self):
        return (
            Object.__getstate__(self),
            self.basis,
            self.origin,
            self.time,
            self.converged,
            self.stable,
            self.second_order,
        )

    def __setstate__(self, serialized):
        Object.__setstate__(self, serialized[0])

        (
            self._basis,
            self._origin,
            self._time,
            self._converged,
            self._stable,
            self._second_order,
        ) = serialized[1:]
