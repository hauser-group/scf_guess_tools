from __future__ import annotations

from ..engine import Engine as Base
from ..molecule import Molecule
from ..wavefunction import Wavefunction


class Engine(Base):
    def __init__(self, cache: bool = True, **kwargs):
        super().__init__(f"{self}" if cache else None, **kwargs)

    def load(self, path: str) -> Molecule:
        from .molecule import Molecule

        return Molecule.load(path)

    @classmethod
    def guess(
        cls, molecule: Molecule, basis: str, scheme: str | None = None
    ) -> Wavefunction:
        from .wavefunction import Wavefunction

        return Wavefunction.guess(molecule, basis, scheme)

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction | None = None
    ) -> Wavefunction:
        from .wavefunction import Wavefunction

        return Wavefunction.calculate(molecule, basis, guess)

    @classmethod
    def __repr__(cls) -> str:
        return "PyEngine"

    @classmethod
    def guessing_schemes(cls) -> list[str]:
        return ["minao", "1e", "atom", "huckel", "vsap"]
