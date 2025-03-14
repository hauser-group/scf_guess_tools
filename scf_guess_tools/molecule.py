from __future__ import annotations

from .core import Object
from abc import ABC, abstractmethod


class Molecule(Object, ABC):
    @property
    @abstractmethod
    def native(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def charge(self) -> int:
        pass

    @property
    @abstractmethod
    def multiplicity(self) -> int:
        pass

    @property
    def singlet(self) -> bool:
        return self.multiplicity == 1

    @property
    @abstractmethod
    def atoms(self) -> int:
        pass

    @property
    @abstractmethod
    def geometry(self):
        pass
