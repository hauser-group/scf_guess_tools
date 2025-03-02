from abc import ABC, abstractmethod
from typing import Self


class Molecule(ABC):
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
    def triplet(self) -> bool:
        return self.multiplicity == 3

    @property
    @abstractmethod
    def atoms(self) -> int:
        pass

    @property
    @abstractmethod
    def geometry(self):
        pass

    def __eq__(self, other: Self) -> bool:
        return (
            self.name == other.name
            and self.charge == other.charge
            and self.multiplicity == other.multiplicity
            and self.atoms == other.atoms
            and self.geometry == other.geometry
        )

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, serialized):
        pass
