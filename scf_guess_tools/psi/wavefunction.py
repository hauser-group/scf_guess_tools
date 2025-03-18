from __future__ import annotations

from ..common import timeable, tuplifyable
from ..wavefunction import Wavefunction as Base
from .auxilary import clean_context
from .core import Object, output_file
from .matrix import Matrix
from .molecule import Molecule
from psi4.core import Wavefunction as Native
from time import process_time

import os
import psi4
import re


class Wavefunction(Base, Object):
    """Wavefunction representation using the Psi4 backend. This class provides an
    implementation of the Wavefunction interface using Psi4's native wavefunction format.
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
        return Matrix(self._native.S())

    @timeable
    def core_hamiltonian(self) -> Matrix:
        """Compute the core Hamiltonian matrix.

        Returns:
            The core Hamiltonian matrix.
        """
        return Matrix(self.native.H())

    @timeable
    @tuplifyable
    def density(self) -> Matrix | tuple[Matrix, Matrix]:
        """Compute the density matrix.

        Returns:
            A single matrix for RHF or a tuple of alpha and beta matrices for UHF.
        """
        if self.molecule.singlet:
            return Matrix(self._native.Da())

        return (Matrix(self._native.Da()), Matrix(self._native.Db()))

    @timeable
    @tuplifyable
    def fock(self) -> Matrix | tuple[Matrix, Matrix]:
        """Compute the Fock matrix.

        Returns:
            A single matrix for RHF or a tuple of alpha and beta matrices for UHF.
        """
        if self.molecule.singlet:
            return Matrix(self._native.Fa())

        return (Matrix(self._native.Fa()), Matrix(self._native.Fb()))

    def __init__(
        self,
        native: Native,
        molecule: Molecule,
        initial: str | Wavefunction,
        *args,
        **kwargs,
    ):
        """Initialize the Psi4 wavefunction. Should not be used directly.

        Args:
            native: The Psi4 Wavefunction instance.
            molecule: The molecule associated with the wavefunction.
            initial: The initial guess, either as scheme or another wavefunction.
            *args: Positional arguments to forward to the base class.
            **kwargs: Keyword arguments to forward to the base class.
        """
        super().__init__(*args, **kwargs)
        self._native = native
        self._molecule = molecule
        self._initial = initial

    def __getstate__(self):
        """Return a serialized representation for pickling.

        Returns:
            The serialized wavefunction object.
        """
        return (
            super().__getstate__(),
            self._native.to_file(),
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
        self._native = Native.from_file(serialized[1])

        self._molecule = Molecule.__new__(Molecule)
        self._molecule.__setstate__(serialized[2])

        if isinstance(serialized[3], str):
            self._initial = serialized[3]
        else:
            self._initial = Wavefunction.__new__(Wavefunction)
            self._initial.__setstate__(serialized[3])

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
        scheme = "AUTO" if scheme is None else scheme

        with clean_context():
            psi4.set_options({"BASIS": basis, "GUESS": scheme})
            basis_set = psi4.core.BasisSet.build(molecule.native, target=basis)
            ref_wfn = psi4.core.Wavefunction.build(molecule.native, basis_set)
            start_wfn = psi4.driver.scf_wavefunction_factory(
                name="HF",
                ref_wfn=ref_wfn,
                reference="RHF" if molecule.singlet else "UHF",
            )

            start_wfn.form_H()
            start_wfn.form_Shalf()
            start_wfn.guess()

            start_wfn.initialize()
            start_wfn.form_G()
            start_wfn.form_F()

            # to_file() different after re-loading with from_file()
            start_wfn = Native.from_file(start_wfn.to_file())

            end = process_time()

            return Wavefunction(
                start_wfn,
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
        attempt to resolve them for the UHF case. Initially try first order HF, then
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

        guess = "AUTO" if guess is None else guess
        guess_str = guess

        if not isinstance(guess, str):
            # https://forum.psicode.org/t/custom-guess-for-hartree-fock/2026/6
            scratch_dir = psi4.core.IOManager.shared_object().get_default_path()
            guess_file = f"{scratch_dir}/stdout.{molecule.name}.{os.getpid()}.180.npy"
            guess.native.to_file(filename=guess_file)
            guess_str = "READ"

        def calculate(second_order: bool, so_max_iterations: int | None = None):
            with clean_context():
                _, wfn = _hartree_fock(
                    molecule, guess_str, basis, second_order, so_max_iterations
                )
                converged, stable = _analyze_output(output_file)
                return wfn, converged, stable

        wfn, converged, stable, second_order = None, False, False, False
        satisfied = lambda: converged and (molecule.singlet or stable)

        try:
            wfn, converged, stable = calculate(second_order)

            if not satisfied():
                raise RuntimeError()
        except:
            second_order = True
            so_max_iterations = 5

            while not satisfied() and so_max_iterations <= 50:
                try:
                    wfn, converged, stable = calculate(second_order, so_max_iterations)
                except psi4.SCFConvergenceError as e:
                    wfn = e.wfn

                so_max_iterations += 5

        end = process_time()

        return Wavefunction(
            wfn,
            molecule,
            guess,
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
) -> tuple[float, Native]:
    options = {
        "BASIS": basis,
        "REFERENCE": "RHF" if molecule.singlet else "UHF",
        "GUESS": guess,
        # Disable density fitting for highest possible accuracy and
        # because stability analysis is not available for density fitted
        # RHF wave functions:
        "SCF_TYPE": "PK",
        "STABILITY_ANALYSIS": "CHECK" if molecule.singlet else "FOLLOW",
    }

    if second_order:
        options["SOSCF"] = True
        options["SOSCF_MAX_ITER"] = so_max_iterations

    psi4.set_options(options)

    return psi4.energy("hf", molecule=molecule.native, return_wfn=True)


def _analyze_output(stdout_file: str) -> tuple[bool, bool]:
    converged, stable = None, None

    with open(stdout_file, "r") as output:
        for line in output:
            line = line.lower()

            if "unable to find file" in line:
                raise RuntimeError(line)
            elif "energy and wave function converged" in line:
                converged = True
            elif "wavefunction unstable" in line:
                stable = False
            elif "wavefunction stable" in line:
                stable = True
            elif "lowest singlet (rhf->rhf) stability eigenvalues:" in line:
                stable = _analyze_stability_eigenvalues(output)

    if converged is None or stable is None:
        raise RuntimeError("Unable to determine convergence and stability")

    return converged, stable


def _analyze_stability_eigenvalues(output) -> bool:
    lines, done = "", False

    for line in output:
        if "lowest triplet (rhf->uhf) stability eigenvalues:" in line.lower():
            done = True
            break

        lines += f"{line}\n"

    if not done:
        raise RuntimeError("Unable to analyze stability eigenvalues")

    pattern = r"-?\d+\.\d+|-?\d+"
    eigenvalues = [float(n) for n in re.findall(pattern, lines)]

    return not any(e < 0.0 for e in eigenvalues)
