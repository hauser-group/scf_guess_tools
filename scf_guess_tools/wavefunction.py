from .molecule import Molecule
from abc import ABC, abstractmethod
from typing import Self


class Wavefunction(ABC):
    def __init__(self, molecule: Molecule, origin: str):
        self._molecule = molecule
        self._origin = origin

    @property
    def molecule(self) -> Molecule:
        return self._molecule

    @property
    def origin(self) -> str:
        return self._origin

    @property
    @abstractmethod
    def native(self):
        pass

    @property
    @abstractmethod
    def Da(self):
        pass

    @property
    @abstractmethod
    def Db(self):
        pass

    @classmethod
    @abstractmethod
    def guess(cls, molecule: Molecule, basis: str, method: str) -> Self:
        pass
