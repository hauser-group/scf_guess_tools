from __future__ import annotations

from ..engine import Engine as Base
from ..metric import Metric
from .metric import f_score, diis_error, energy_error
from .molecule import Molecule
from .wavefunction import Wavefunction


class Engine(Base):
    def __init__(self, cache: bool = True, verbose: int = 0):
        super().__init__("psi4" if cache else None, verbose)

    @classmethod
    def backend(cls) -> list[str]:
        return "Psi4"

    @classmethod
    def guessing_schemes(cls) -> list[str]:
        return ["CORE", "SAD", "SADNO", "GWH", "HUCKEL", "MODHUCKEL", "SAP", "SAPGAU"]

    def load(self, path: str) -> Molecule:
        return Molecule(path)

    def guess(self, molecule: Molecule, basis: str, scheme: str) -> Wavefunction:
        return Wavefunction.guess(molecule, basis, scheme)

    def calculate(
        self, molecule: Molecule, basis: str, guess: str | Wavefunction = None
    ) -> Wavefunction:
        return Wavefunction.calculate(molecule, basis, guess)

    def score(
        self, initial: Wavefunction, final: Wavefunction, metric: Metric
    ) -> float:
        if metric == Metric.F_SCORE:
            return f_score(initial, final)
        elif metric == Metric.DIIS_ERROR:
            return diis_error(initial, final)
        elif metric == Metric.ENERGY_ERROR:
            return energy_error(initial, final)
        else:
            raise NotImplementedError(f"{metric} not implemented for Psi4")
