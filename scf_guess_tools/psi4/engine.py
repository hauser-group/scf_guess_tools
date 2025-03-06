from __future__ import annotations

from ..engine import Engine as Base
from tempfile import TemporaryDirectory

import psi4
import scf_guess_tools.psi4.molecule as m
import scf_guess_tools.psi4.wavefunction as w


class Engine(Base):
    def __init__(self, cache: bool = True, verbose: int = 0):
        super().__init__("psi4" if cache else None, verbose)

        self._output_directory = TemporaryDirectory()

        psi4.set_output_file(
            self.output_file, append=False, execute=True, print_header=False
        )

    @property
    def output_file(self) -> str:
        return f"{self._output_directory.name}/stdout"  # don't rename, bug with READ option

    @classmethod
    def __repr__(cls) -> str:
        return "Psi4Engine"

    @classmethod
    def guessing_schemes(cls) -> list[str]:
        return ["CORE", "SAD", "SADNO", "GWH", "HUCKEL", "MODHUCKEL", "SAP", "SAPGAU"]

    def load(self, path: str) -> m.Molecule:
        return m.Molecule.load(self, path)

    def guess(self, molecule: m.Molecule, basis: str, scheme: str) -> w.Wavefunction:
        return w.Wavefunction.guess(self, molecule, basis, scheme)

    def calculate(
        self,
        molecule: m.Molecule,
        basis: str,
        guess: str | w.Wavefunction | None = None,
    ) -> w.Wavefunction:
        return w.Wavefunction.calculate(self, molecule, basis, guess)
