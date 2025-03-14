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
        native = self._native

        if self._is_guess:
            try:
                with clean_context():
                    psi4.set_options({"MAXITER": 0, "FAIL_ON_MAXITER": True})
                    _, native = _hartree_fock(
                        self.molecule, self.initial, self.basis, False
                    )
            except psi4.driver.SCFConvergenceError as e:
                native = e.wfn

        if self.molecule.singlet:
            return Matrix(native.Fa())

        return (Matrix(native.Fa()), Matrix(native.Fb()))

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

            # Calling form_F doesn't deliver a correct fock matrix, so we handle it in self.F
            # We can't handle it here because this would cause a deviation in the density

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
            guess_file = (
                f"{scratch_dir}/" f"stdout.{molecule.name}.{os.getpid()}.180.npy"
            )
            guess.native.to_file(filename=guess_file)
            guess_str = "READ"

        def calculate():
            with clean_context():
                _, wfn = _hartree_fock(molecule, guess_str, basis, retry)
                iterations = _hartree_fock_iterations(output_file)
                return wfn, iterations

        iterations, retry, converged = None, False, True

        try:
            wfn, iterations = calculate()
        except psi4.ConvergenceError:
            retry = True
            try:
                wfn, iterations = calculate()
            except psi4.SCFConvergenceError as e:
                converged = False
                wfn = e.wfn

        end = process_time()

        return Wavefunction(
            wfn,
            False,
            molecule=molecule,
            basis=basis,
            initial=guess,
            origin="calculation",
            time=end - start,
            iterations=iterations,
            retried=retry,
            converged=converged,
        )


def _hartree_fock(
    molecule: Molecule, guess: str, basis: str, second_order: bool = False
) -> tuple[float, Native]:
    stability_analysis = "NONE"
    if not molecule.singlet:
        stability_analysis = "FOLLOW"
    elif molecule.atoms <= 30:
        stability_analysis = "CHECK"

    psi4.set_options(
        {
            "BASIS": basis,
            "REFERENCE": "RHF" if molecule.singlet else "UHF",
            "GUESS": guess,
            # Disable density fitting for highest possible accuracy and
            # because stability analysis is not available for density fitted
            # RHF wave functions:
            "SCF_TYPE": "PK",
            "STABILITY_ANALYSIS": stability_analysis,
        }
    )

    if second_order:
        psi4.set_options(
            {
                "SOSCF": True,
                "SOSCF_START_CONVERGENCE": 1.0e-2,
                "SOSCF_MAX_ITER": 40,
            }
        )

    return psi4.energy("HF", molecule=molecule.native, return_wfn=True)


def _hartree_fock_iterations(stdout_file: str) -> int:
    iterations = []

    with open(stdout_file, "r") as stdout:
        pattern = re.compile(r"^\S+\s+iter\s+(\d+):")
        count = False

        for line in stdout:
            if "Unable to find file" in line:
                raise RuntimeError(line)

            if "Failed to converge" in line:
                count = False
                iterations = []
                continue

            if line.strip() == "==> Iterations <==":
                count = True
                continue

            if count and line.strip().startswith("==>"):
                count = False
                continue

            if count:
                iteration = pattern.match(line.strip())
                if iteration is not None:
                    iterations.append(int(iteration.group(1)))

    # offset = None
    #
    # for i, iteration in enumerate(iterations):
    #     if offset is None:
    #         offset = i - iteration
    #
    #     if i != iteration - 1:
    #         raise RuntimeError(f"extracted invalid iteration sequence {iterations}")

    return len(iterations)
