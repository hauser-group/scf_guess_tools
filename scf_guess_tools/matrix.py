from __future__ import annotations

from abc import ABC, abstractmethod


class Matrix(ABC):
    @property
    @abstractmethod
    def native(self):
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: Matrix) -> bool:
        pass

    @abstractmethod
    def __add__(self, other: Matrix) -> Matrix:
        pass

    @abstractmethod
    def __sub__(self, other: Matrix) -> Matrix:
        pass

    @abstractmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        pass

    @property
    @abstractmethod
    def size(self) -> float:
        pass

    @property
    @abstractmethod
    def trace(self) -> float:
        pass

    @property
    @abstractmethod
    def sum_of_squares(self) -> float:
        pass
