from ..engine import Engine as Base
from .molecule import Molecule
from .wavefunction import Wavefunction


class Engine(Base):
    def load(self, path: str) -> Molecule:
        return Molecule(path)

    def guess(self, molecule: Molecule, basis: str, method: str) \
            -> Wavefunction:
        return Wavefunction.guess(molecule, basis, method)
