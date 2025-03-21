from __future__ import annotations

from ..common import cache, timeable, tuplifyable
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

    def _dft_electronic_energy(self) -> float:
        """Compute the electronic energy for DFT calculations."""
        e_nuc = self.molecule.native.nuclear_repulsion_energy()
        e_total = (
            self.e_total if self.e_total is not None else self.native.energy()
        )  #! TODO: This performs a new calculation!
        return e_total - e_nuc

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
    @timeable
    @cache(enable=False, ignore=["cls"])
    def guess(
        cls,
        molecule: Molecule,
        basis: str,
        scheme: str | None = None,
        method: str = "hf",
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
                method=method,
            )

    @classmethod
    @timeable
    @cache(enable=False, ignore=["cls"])
    def calculate(
        cls,
        molecule: Molecule,
        basis: str,
        guess: str | Wavefunction | None = None,
        method: str = "hf",
        functional: str | None = None,
    ) -> Wavefunction:
        """Attempt to compute a converged wavefunction. Detect instabilities and
        attempt to resolve them for the UHF case. Initially try first order HF, then
        switch to second order HF with increasing number of microiterations until a
        wavefunction is found to be both converged and stable.

        Args:
            molecule: The molecule for which the wavefunction is computed.
            basis: The basis set.
            method (optional): The calculation method to use (hf, dft)
            guess (optional): The initial guess, either as guessing scheme or another wavefunction.
            functional (optional): The functional to use for dft calculations

        Returns:
            The computed wavefunction.
        """
        start = process_time()
        guess = "AUTO" if guess is None else guess

        def calculate(second_order: bool, so_max_iterations: int | None = None):
            with clean_context():
                guess_str = guess

                if not isinstance(guess, str):
                    # https://forum.psicode.org/t/custom-guess-for-hartree-fock/2026/6
                    scratch = psi4.core.IOManager.shared_object().get_default_path()
                    guess_file = (
                        f"{scratch}/stdout.{molecule.name}.{os.getpid()}.180.npy"
                    )
                    guess.native.to_file(filename=guess_file)
                    guess_str = "READ"

                e_total, wfn = _scf_calculation(
                    molecule,
                    guess_str,
                    basis,
                    method,
                    functional,
                    second_order,
                    so_max_iterations,
                )
                converged, stable = _analyze_output(output_file, method)

                return wfn, converged, stable, e_total

        wfn, converged, stable, second_order = None, False, False, False
        satisfied = lambda: converged and (molecule.singlet or stable)

        try:
            wfn, converged, stable, e_total = calculate(second_order)

            if not satisfied():
                raise RuntimeError()
        except (
            psi4.driver.p4util.exceptions.ValidationError
        ) as e:  # method might not support Validation
            print(f"Validation Error: {e} - ")

        except:
            second_order = True
            so_max_iterations = 5

            if method != "dft":  # no second order corrections implemented for DFT
                while not satisfied() and so_max_iterations <= 50:
                    try:
                        wfn, converged, stable, e_total = calculate(
                            second_order, so_max_iterations
                        )
                    except psi4.SCFConvergenceError as e:
                        wfn = e.wfn

                    so_max_iterations += 5

        end = process_time()

        return Wavefunction(
            wfn,
            molecule,
            guess,
            basis=basis,
            method=method,
            functional=functional,
            origin="calculation",
            time=end - start,
            converged=converged,
            stable=stable,
            second_order=(
                second_order if method == "hf" else None
            ),  # second order not implemented for DFT
            e_total=e_total,
        )


def _scf_calculation(
    molecule: Molecule,
    guess: str,
    basis: str,
    method: str = "hf",
    functional: str | None = None,
    second_order: bool = False,
    so_max_iterations: int | None = None,
) -> tuple[float, Native]:
    """
    Run an SCF calculation (Hartree-Fock or DFT) using Psi4.

    Args:
        molecule: The molecule to compute.
        guess: Initial guess method.
        basis: Basis set.
        method: "hf" for Hartree-Fock, "dft" for Density Functional Theory.
        functional: The exchange-correlation functional (only for DFT).
        second_order: Enable second-order SCF.
        so_max_iterations: Maximum iterations for second-order SCF.

    Returns:
        A tuple (energy, wavefunction).
    """
    method = method.lower()
    if method not in ("hf", "dft"):
        raise ValueError(f"Unsupported method: {method}")

    options = {
        "BASIS": basis,
        "GUESS": guess,
        "SCF_TYPE": "PK",
    }

    if method == "dft":
        if functional is None:
            functional = "B3LYP"  # Default to B3LYP if not provided
            import warnings

            warnings.warn(
                "DFT functional was not provided. Defaulting to 'B3LYP'.", UserWarning
            )
        options["REFERENCE"] = "RKS" if molecule.singlet else "UKS"
        #! maybe add later
        # options["DFT_SPHERICAL_POINTS"] = 590
        # options["DFT_RADIAL_POINTS"] = 99

    elif method == "hf":
        options["REFERENCE"] = "RHF" if molecule.singlet else "UHF"
        options["STABILITY_ANALYSIS"] = "CHECK" if molecule.singlet else "FOLLOW"

    if second_order:
        options["SOSCF"] = True
        options["SOSCF_MAX_ITER"] = so_max_iterations

    psi4.set_options(options)

    if method == "dft":
        return psi4.energy(
            functional,
            molecule=molecule.native,
            return_wfn=True,
        )

    return psi4.energy(
        "hf",
        molecule=molecule.native,
        return_wfn=True,
    )


def _analyze_output(stdout_file: str, method: str) -> tuple[bool, bool]:
    converged, stable = None, None

    with open(stdout_file, "r") as output:
        output_lines = output.readlines()

        for line in output_lines:
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
                stable = _analyze_stability_eigenvalues(output_lines)

    if converged is None or stable is None:
        if method == "hf":
            raise RuntimeError("Unable to determine convergence or stability for HF")
        elif method == "dft" and converged is None:
            raise RuntimeError("Unable to determine convergence for DFT")

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
