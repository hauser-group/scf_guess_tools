from __future__ import annotations

from .core import Object
from abc import ABC, abstractmethod
from numpy.typing import NDArray


class Matrix(Object, ABC):
    @property
    @abstractmethod
    def native(self):
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

    @property
    @abstractmethod
    def numpy(self) -> NDArray:
        pass

    @abstractmethod
    def __repr__(self) -> str:
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

    @classmethod
    @abstractmethod
    def build(cls, array: NDArray) -> Matrix:
        pass
