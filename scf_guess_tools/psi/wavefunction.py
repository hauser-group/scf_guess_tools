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
    @property
    def native(self) -> Native:
        return self._native

    @timeable
    def overlap(self) -> Matrix:
        return Matrix(self._native.S())

    @timeable
    def core_hamiltonian(self) -> Matrix:
        return Matrix(self.native.H())

    @timeable
    @tuplifyable
    def density(self) -> Matrix | tuple[Matrix, Matrix]:
        if self.molecule.singlet:
            return Matrix(self._native.Da())

        return (Matrix(self._native.Da()), Matrix(self._native.Db()))

    @timeable
    @tuplifyable
    def fock(self) -> Matrix | tuple[Matrix, Matrix]:
        if self.molecule.singlet:
            return Matrix(self._native.Fa())

        return (Matrix(self._native.Fa()), Matrix(self._native.Fb()))

    def __init__(self, native: Native, is_guess: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native = native
        self._is_guess = is_guess

    def __getstate__(self):
        return super().__getstate__(), self.native.to_file(), self._is_guess

    def __setstate__(self, serialized):
        super().__setstate__(serialized[0])
        self._native = Native.from_file(serialized[1])
        self._is_guess = serialized[2]

    @classmethod
    def guess(
        cls, molecule: Molecule, basis: str, scheme: str | None = None
    ) -> Wavefunction:
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
                True,
                molecule=molecule,
                basis=basis,
                initial=scheme,
                origin="guess",
                time=end - start,
            )

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction | None = None
    ) -> Wavefunction:
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
            False,
            molecule=molecule,
            basis=basis,
            initial=guess,
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
