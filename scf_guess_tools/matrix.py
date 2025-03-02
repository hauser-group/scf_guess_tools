from __future__ import annotations

from abc import ABC, abstractmethod


class Matrix(ABC):
    @property
    @abstractmethod
    def native(self):
        pass

    @abstractmethod
    def __eq__(self, other: Matrix) -> bool:
        pass
