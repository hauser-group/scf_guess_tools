from __future__ import annotations

from ..engine import Engine as Base
from ..molecule import Molecule
from ..wavefunction import Wavefunction
from tempfile import TemporaryDirectory

import psi4


class Engine(Base):
    def __init__(self, cache: bool = True, verbose: int = 0, **kwargs):
        super().__init__(f"{self}" if cache else None, verbose)

        self._output_directory = TemporaryDirectory()

        psi4.set_output_file(
            self.output_file, append=False, execute=True, print_header=False
        )

    @property
    def output_file(self) -> str:
        return f"{self._output_directory.name}/stdout"  # don't rename, bug with READ option

    def load(self, path: str) -> Molecule:
        from .molecule import Molecule

        return Molecule.load(path)

    def guess(self, molecule: Molecule, basis: str, scheme: str) -> Wavefunction:
        from .wavefunction import Wavefunction

        return Wavefunction.guess(molecule, basis, scheme)

    def calculate(
        self, molecule: Molecule, basis: str, guess: str | Wavefunction | None = None
    ) -> Wavefunction:
        from .wavefunction import Wavefunction

        return Wavefunction.calculate(molecule, basis, guess)

    @classmethod
    def __repr__(cls) -> str:
        return "PsiEngine"

    @classmethod
    def guessing_schemes(cls) -> list[str]:
        return ["CORE", "SAD", "SADNO", "GWH", "HUCKEL", "MODHUCKEL", "SAP", "SAPGAU"]
