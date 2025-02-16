from ..metric import Metric
from ..engine import Engine as Base
from .metric import f_score, diis_error, energy_error
from .molecule import Molecule
from .wavefunction import Wavefunction


class Engine(Base):
    def __init__(self, cache: bool = True, verbose: int = 0):
        super().__init__("pyscf" if cache else None, verbose)

    def load(self, path: str) -> Molecule:
        return Molecule(path)

    def guess(self, molecule: Molecule, basis: str, method: str) -> Wavefunction:
        return Wavefunction.guess(molecule, basis, method)

    def calculate(
        self, molecule: Molecule, basis: str, guess: str | Wavefunction = None
    ) -> Wavefunction:
        return Wavefunction.calculate(molecule, basis, guess)

    def score(
        self, initial: Wavefunction, final: Wavefunction, metric: Metric
    ) -> float:
        match metric:
            case Metric.F_SCORE:
                return f_score(initial, final)
            case Metric.DIIS_ERROR:
                return diis_error(initial, final)
            case Metric.ENERGY_ERROR:
                return energy_error(initial, final)
            case _:
                raise NotImplementedError(f"{metric} not implemented for PySCF")

    @property
    def guessing_schemes(self) -> list[str]:
        return ["minao", "1e", "atom", "huckel", "vsap"]
