from abc import ABC, abstractmethod


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

    @abstractmethod
    def __getstate__(self):
        pass

    @abstractmethod
    def __setstate__(self, serialized):
        pass
