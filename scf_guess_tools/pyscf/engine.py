from ..engine import Engine as Base
from .molecule import Molecule
from .wavefunction import Wavefunction


class Engine(Base):
    def load(self, path: str) -> Molecule:
        return Molecule(path)

    @classmethod
    def guess(cls, molecule: Molecule, basis: str,
              method: str) -> Wavefunction:
        return Wavefunction.guess(molecule, basis, method)

    @classmethod
    def calculate(cls, molecule: Molecule, basis: str,
                  guess: str | Wavefunction) -> Wavefunction:
        return Wavefunction.calculate(molecule, basis, guess)
