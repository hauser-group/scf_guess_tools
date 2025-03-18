from __future__ import annotations


from ..common import timeable, tuplifyable
from ..wavefunction import Wavefunction as Base
from .core import Object, guessing_schemes
from .matrix import Matrix
from .molecule import Molecule
from numpy.typing import NDArray
from pyscf.scf import RHF, UHF
from pyscf.scf.hf import SCF as Native
from time import process_time


class Wavefunction(Base, Object):
    """Wavefunction representation using the PySCF backend. This class provides an
    implementation of the Wavefunction interface using PySCF's native wavefunction format.
    """

    @property
    def native(self) -> Native:
        """The underlying Psi4 wavefunction object."""
        return self._native

    @property
    def molecule(self) -> Molecule:
        """The molecule associated with this wavefunction."""
        return self._molecule

    @property
    def initial(self) -> str | Wavefunction:
        """The initial guess, either as a string (guessing scheme) or another
        wavefunction."""
        return self._initial

    @timeable
    def overlap(self) -> Matrix:
        """Compute the overlap matrix.

        Returns:
            The overlap matrix.
        """
        return Matrix(self._native.get_ovlp())

    @timeable
    def core_hamiltonian(self) -> Matrix:
        """Compute the core Hamiltonian matrix.

        Returns:
            The core Hamiltonian matrix.
        """
        return Matrix(self.native.get_hcore())

    @timeable
    @tuplifyable
    def density(self) -> Matrix | tuple[Matrix, Matrix]:
        """Compute the density matrix.

        Returns:
            A single matrix for RHF or a tuple of alpha and beta matrices for UHF.
        """
        if self.molecule.singlet:
            return Matrix(self._D / 2)

        return Matrix(self._D[0]), Matrix(self._D[1])

    @timeable
    @tuplifyable
    def fock(self) -> Matrix | tuple[Matrix, Matrix]:
        """Compute the Fock matrix.

        Returns:
            A single matrix for RHF or a tuple of alpha and beta matrices for UHF.
        """

        F = self._native.get_fock(dm=self._D)

        if self.molecule.singlet:
            return Matrix(F)

        return Matrix(F[0]), Matrix(F[1])

    def __init__(
        self,
        native: Native,
        D: NDArray,
        molecule: Molecule,
        initial: str | Wavefunction,
        *args,
        **kwargs,
    ):
        """Initialize the PySCF wavefunction. Should not be used directly.

        Args:
            native: The PySCF solver instance.
            D: The density matrix.
            molecule: The molecule associated with the wavefunction.
            initial: The initial guess, either as scheme or another wavefunction.
            *args: Positional arguments to forward to the base class.
            **kwargs: Keyword arguments to forward to the base class.
        """
        super().__init__(*args, **kwargs)
        self._native = native
        self._D = D
        self._molecule = molecule
        self._initial = initial

    def __getstate__(self):
        """Return a serialized representation for pickling.

        Returns:
            The serialized wavefunction object.
        """
        return (
            super().__getstate__(),
            self._D,
            self._molecule.__getstate__(),
            (
                self._initial.__getstate__()
                if isinstance(self._initial, Wavefunction)
                else self._initial
            ),
        )

    def __setstate__(self, serialized):
        """Restore the wavefunction from a serialized state.

        Args:
            serialized: The serialized wavefunction object.
        """
        super().__setstate__(serialized[0])
        self._D = serialized[1]

        self._molecule = Molecule.__new__(Molecule)
        self._molecule.__setstate__(serialized[2])

        if isinstance(serialized[3], str):
            self._initial = serialized[3]
        else:
            self._initial = Wavefunction.__new__(Wavefunction)
            self._initial.__setstate__(serialized[3])

        self._molecule.native.basis = self._basis
        self._molecule.native.build()

        method = RHF if self._molecule.singlet else UHF
        self._native = method(self._molecule.native)

    @classmethod
    def guess(
        cls, molecule: Molecule, basis: str, scheme: str | None = None
    ) -> Wavefunction:
        """Create an initial wavefunction guess.

        Args:
            molecule: The molecule for which the wavefunction is created.
            basis: The basis set.
            scheme: The initial guess scheme. If None, the default scheme is used.

        Returns:
            The guessed wavefunction.
        """

        start = process_time()

        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        solver = method(molecule.native)

        scheme = solver.init_guess if scheme is None else scheme

        D = solver.get_init_guess(key=scheme)
        end = process_time()

        return Wavefunction(
            solver,
            D,
            molecule,
            scheme,
            basis=basis,
            origin="guess",
            time=end - start,
        )

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction | None = None
    ) -> Wavefunction:
        """Attempt to compute a converged wavefunction. Detect instabilities and
        attempt to resolve them for both RHF and UHF. Initially try first order HF, then
        switch to second order HF with increasing number of microiterations until a
        wavefunction is found to be both converged and stable.

        Args:
            molecule: The molecule for which the wavefunction is computed.
            basis: The basis set.
            guess: The initial guess, either as guessing scheme or another wavefunction.

        Returns:
            The computed wavefunction.
        """
        start = process_time()

        molecule.native.basis = basis
        molecule.native.build()

        def calculate(second_order: bool, so_max_iterations: int | None = None):
            return _hartree_fock(
                molecule, guess, basis, second_order, so_max_iterations
            )

        second_order = False
        solver, converged, stable = calculate(second_order)
        satisfied = lambda: converged and (molecule.singlet or stable)

        if not satisfied():
            second_order = True
            so_max_iterations = 5

            while not satisfied() and so_max_iterations <= 50:
                solver, converged, stable = calculate(second_order, so_max_iterations)
                so_max_iterations += 5

        end = process_time()

        return Wavefunction(
            solver,
            solver.make_rdm1(),
            molecule,
            solver.init_guess if guess is None else guess,
            basis=basis,
            origin="calculation",
            time=end - start,
            converged=converged,
            stable=stable,
            second_order=second_order,
        )


def _hartree_fock(
    molecule: Molecule,
    guess: str,
    basis: str,
    second_order: bool,
    so_max_iterations: int | None,
) -> tuple[Native, bool, bool]:
    method = RHF if molecule.singlet else UHF
    solver = method(molecule.native)

    if second_order:
        solver = solver.newton()
        solver.max_cycle_inner = so_max_iterations

    guess = solver.init_guess if guess is None else guess

    if isinstance(guess, str):
        assert guess in guessing_schemes
        solver.run(init_guess=guess)
    else:
        solver.kernel(dm0=guess._D)

    converged, stable = solver.converged, False

    if converged:
        stability_options = {
            "internal": True,
            "external": False,
            "return_status": True,
        }

        mo, _, stable, _ = solver.stability(**stability_options)
        retries = 0

        while not molecule.singlet and not stable and retries < 15:
            dm = solver.make_rdm1(mo, solver.mo_occ)
            solver = solver.run(dm)
            mo, _, stable, _ = solver.stability(**stability_options)
            retries += 1

    return solver, converged, stable
