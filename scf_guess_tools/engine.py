from .molecule import Molecule
from abc import ABC, abstractmethod


class Engine(ABC):
    @abstractmethod
    def load(self, path: str) -> Molecule:
        pass
