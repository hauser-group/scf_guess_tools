from ..engine import Engine as Base
from .molecule import Molecule


class Engine(Base):
    def load(self, path: str) -> Molecule:
        return Molecule(path)
