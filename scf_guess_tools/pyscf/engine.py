from ..metric import Metric
from ..engine import Engine as Base
from .metric import f_score, diis_error, energy_error
from .molecule import Molecule
from .wavefunction import Wavefunction


class Engine(Base):
    def load(self, path: str) -> Molecule:
        return Molecule(path)

    @classmethod
    def guess(cls, molecule: Molecule, basis: str, method: str) -> Wavefunction:
        return Wavefunction.guess(molecule, basis, method)

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction
    ) -> Wavefunction:
        return Wavefunction.calculate(molecule, basis, guess)

    @classmethod
    def score(cls, initial: Wavefunction, final: Wavefunction, metric: Metric) -> float:
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
        return ["minao", "1e", "atom", "huckel", "vsap", "chk"]
