from abc import ABC, abstractmethod
from typing import Self


class Matrix(ABC):
    @property
    @abstractmethod
    def native(self):
        pass

    @abstractmethod
    def __eq__(self, other: Self) -> bool:
        pass
