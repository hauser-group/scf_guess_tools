from __future__ import annotations

from .common import cache, timeable, tuplifyable
from .core import Object
from .matrix import Matrix
from .molecule import Molecule
from .proxy import proxy
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any

import joblib


class Wavefunction(Object, ABC):
    """Provides a common interface for wavefunctions across different backends."""

    @property
    @abstractmethod
    def native(self) -> Any:
        """The underlying backend-specific wavefunction object."""
        pass

    @property
    @abstractmethod
    def molecule(self) -> Molecule:
        """The molecule associated with this wavefunction."""
        pass

    @property
    def basis(self) -> str:
        """The basis set used to construct the wavefunction."""
        return self._basis

    @property
    @abstractmethod
    def initial(self) -> str | Wavefunction:
        """The initial guess, either as a string (guessing scheme) or another
        wavefunction."""
        pass

    @property
    def origin(self) -> str:
        """Describes how the wavefunction was created (e.g. via guess or calculate)."""
        return self._origin

    @property
    def time(self) -> float:
        """Time required to build or compute the wavefunction."""
        return self._time

    @property
    def converged(self) -> bool | None:
        """Whether the wavefunction calculation converged. Not applicable to guesses."""
        return self._converged

    @property
    def stable(self) -> bool | None:
        """Whether the wavefunction is stable. Not applicable to guesses."""
        return self._stable

    @property
    def second_order(self) -> bool | None:
        """Whether second-order SCF was required due to non-convergence or internal
        instabilities."""
        return self._second_order

    @property
    def functional(self) -> str | None:
        """The functional used if the calculation was started with method=dft (e.g., 'PBE', 'B3LYP')"""
        return self._functional

    @property
    def method(self) -> str | None:
        """The method used to calculate the wavefunction ('hf' or 'dft')"""
        return self._method

    @property
    def e_total(self) -> float:
        """The total energy of the wavefunction."""
        return self._e_total

    @abstractmethod
    @timeable
    def overlap(self) -> Matrix:
        """Compute the overlap matrix.

        Returns:
            The overlap matrix.
        """
        pass

    @abstractmethod
    @timeable
    def core_hamiltonian(self) -> Matrix:
        """Compute the core Hamiltonian matrix.

        Returns:
            The core Hamiltonian matrix.
        """
        pass

    @abstractmethod
    @timeable
    @tuplifyable
    def density(self) -> Matrix | tuple[Matrix, Matrix]:
        """Compute the density matrix.

        Returns:
            A single matrix for RHF or a tuple of alpha and beta matrices for UHF.
        """
        pass

    @abstractmethod
    @timeable
    @tuplifyable
    def fock(self) -> Matrix | tuple[Matrix, Matrix]:
        """Compute the Fock matrix.

        Returns:
            A single matrix for RHF or a tuple of alpha and beta matrices for UHF.
        """
        pass

    def nuclear_repulsion_energy(self) -> float:
        """Compute the nuclear repulsion energy.

        Returns:
            The computed nuclear repulsion energy.
        """
        pass

    @tuplifyable
    def electronic_energy(
        self,
        core_hamiltonian: Matrix | None = None,
        density: Matrix | tuple[Matrix, Matrix] = None,
        fock: Matrix | tuple[Matrix, Matrix] | None = None,
    ) -> float:
        """Compute the electronic energy.

        Args:
            core_hamiltonian: The core Hamiltonian matrix. If None,
                self.core_hamiltonian() is used.
            density: A single density matrix for RHF or a tuple of alpha and beta
                matrices for UHF. If None, self.density() is used.
            fock: A single Fock matrix for RHF or a tuple of alpha and beta matrices for
                UHF. If None, self.fock() is used.

        Returns:
            The computed electronic energy.
        """
        H = self.core_hamiltonian() if core_hamiltonian is None else core_hamiltonian
        D = self.density() if density is None else density
        F = self.fock() if fock is None else fock

        if self.method == "dft":
            return float(self._dft_electronic_energy())

        if self.molecule.singlet:
            # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3a_restricted-hartree-fock.ipynb

            result = (F + H) @ D
            return float(result.trace)
        else:
            # https://github.com/psi4/psi4numpy/blob/master/Tutorials/03_Hartree-Fock/3c_unrestricted-hartree-fock.ipynb

            Da, Db = D
            Fa, Fb = F

            terms = [(a @ b).trace for a, b in zip([Da + Db, Da, Db], [H, Fa, Fb])]
            return float(0.5 * sum(terms))

    def __init__(
        self,
        basis: str,
        origin: str,
        time: float,
        converged: bool | None = None,
        stable: bool | None = None,
        second_order: bool | None = None,
        functional: str | None = None,
        method: str | None = None,
        e_total: float | None = None,
    ):
        """Initialize the wavefunction. Should not be used directly.

        Args:
            basis: The basis set used.
            origin: How the wavefunction was created (e.g. via guess or calculate).
            time: Time required for computation.
            converged: Whether the calculation converged (if applicable).
            stable: Whether the wavefunction is stable (if applicable).
            second_order: Whether second-order SCF was needed due to convergence
                issues or instabilities.
            functional: The functional used for DFT calculations.
            method: The calculation method used (hf, dft).
            e_total: The total energy of the wavefunction
        """
        self._basis = basis
        self._origin = origin
        self._time = time
        self._converged = converged
        self._stable = stable
        self._second_order = second_order
        self._functional = functional
        self._method = method
        self._e_total = e_total

    def __hash__(self) -> int:
        """Return a deterministic hash.

        Returns:
            A hash value uniquely identifying the wavefunction.
        """
        identity = (
            self.backend(),
            self.molecule.__hash__(),
            self.basis,
            (
                self.initial.__hash__()
                if isinstance(self.initial, Object)
                else joblib.hash(self.initial)
            ),
            self.origin,
            self.converged,
            self.stable,
            self.second_order,
            self.functional,
            self.method,
            self.e_total,
        )

        return int(joblib.hash(identity), 16)

    def __getstate__(self):
        """Return a serialized representation for pickling.

        Returns:
            The serialized wavefunction object.
        """
        return (
            Object.__getstate__(self),
            self.basis,
            self.origin,
            self.time,
            self.converged,
            self.stable,
            self.second_order,
            self.method,
            self.functional,
            self.e_total,
        )

    def __setstate__(self, serialized):
        """Restore the wavefunction from a serialized state.

        Args:
            serialized: The serialized wavefunction object.
        """
        Object.__setstate__(self, serialized[0])

        (
            self._basis,
            self._origin,
            self._time,
            self._converged,
            self._stable,
            self._second_order,
            self._method,
            self._functional,
            self._e_total,
        ) = serialized[1:]

    @classmethod
    @abstractmethod
    @timeable
    @cache(enable=False, ignore=["cls"])
    def guess(
        cls,
        molecule: Molecule,
        basis: str,
        scheme: str | None = None,
        method: str = "hf",
        **kwargs,
    ) -> Wavefunction:
        """Create an initial wavefunction guess. For backend-specific behavior please
        refer to the docstrings of the actual implemention.

        Args:
            molecule: The molecule for which the wavefunction is created.
            basis: The basis set.
            scheme: The initial guess scheme. If None, the default scheme is used.
            method: The calculation method to use (hf, dft)
            **kwargs: Additional backend-specific keyword arguments.

        Returns:
            The guessed wavefunction.
        """
        pass

    @classmethod
    @abstractmethod
    @timeable
    @cache(enable=False, ignore=["cls"])
    def calculate(
        cls,
        molecule: Molecule,
        basis: str,
        guess: str | Wavefunction | None = None,
        method: str = "hf",
        functional: str | None = None,
        **kwargs,
    ) -> Wavefunction:
        """Attempt to compute a converged wavefunction. For backend-specific behavior please
        refer to the docstrings of the actual implemention.

        Args:
            molecule: The molecule for which the wavefunction is computed.
            basis: The basis set.
            guess: The initial guess, either as guessing scheme or another wavefunction.
            method: The calculation method to use (hf, dft)
            functional: The functional to use for dft calculations
            **kwargs: Additional backend-specific keyword arguments.

        Returns:
            The computed wavefunction.
        """
        start = process_time()


@wraps(Wavefunction.guess)
def guess(molecule: Molecule, *args, **kwargs):
    return proxy(
        molecule.backend(), lambda p: p.Wavefunction.guess, molecule, *args, **kwargs
    )


@wraps(Wavefunction.calculate)
def calculate(molecule: Molecule, *args, **kwargs):
    return proxy(
        molecule.backend(),
        lambda p: p.Wavefunction.calculate,
        molecule,
        *args,
        **kwargs,
    )
