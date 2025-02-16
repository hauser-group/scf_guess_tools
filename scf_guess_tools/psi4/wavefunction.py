import os
import psi4
import re

from ..wavefunction import Wavefunction as Base
from .auxilary import clean_context
from .molecule import Molecule
from psi4.core import Matrix, Wavefunction as Native
from typing import Self


def _hartree_fock(
    molecule: Molecule, guess: str, basis: str, second_order: bool = False
) -> Native:
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

    _, wfn = psi4.energy("HF", molecule=molecule.native, return_wfn=True)
    return wfn


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


class Wavefunction(Base):
    def __init__(self, native: Native, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native = native

    @property
    def native(self) -> Native:
        return self._native

    @property
    def Da(self) -> Matrix:
        return self._native.Da_subset("AO")

    @property
    def Db(self) -> Matrix:
        return self._native.Db_subset("AO")

    @classmethod
    def guess(cls, molecule: Molecule, basis: str, method: str) -> Self:
        with clean_context():
            psi4.set_options({"BASIS": basis, "GUESS": method})

            basis = psi4.core.BasisSet.build(molecule.native, target=basis)
            ref_wfn = psi4.core.Wavefunction.build(molecule.native, basis)
            start_wfn = psi4.driver.scf_wavefunction_factory(
                name="HF",
                ref_wfn=ref_wfn,
                reference="RHF" if molecule.singlet else "UHF",
            )
            start_wfn.form_H()
            start_wfn.form_Shalf()
            start_wfn.guess()

            return Wavefunction(start_wfn, molecule, method)

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Self = None
    ) -> Self:
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

        try:
            with clean_context() as stdout_file:
                retry = False
                wfn = _hartree_fock(molecule, guess_str, basis, retry)
                iterations = _hartree_fock_iterations(stdout_file)
        except psi4.ConvergenceError:
            with clean_context() as stdout_file:
                retry = True
                wfn = _hartree_fock(molecule, guess_str, basis, retry)
                iterations = _hartree_fock_iterations(stdout_file)

        return Wavefunction(wfn, molecule, guess, iterations, retry)

    def __getstate__(self):
        return (self.molecule,
                self.initial.native.to_file() if isinstance(self.initial, Native) else self.initial,
                self.iterations,
                self.retried,
                self.native.to_file()
                )

    def __setstate__(self, serialized):
        self._molecule = serialized[0]
        self._initial = serialized[1] if isinstance(serialized[1], str) else Native.from_file(serialized[1])
        self._iterations = serialized[2]
        self._retried = serialized[3]
        self._native = Native.from_file(serialized[4])
