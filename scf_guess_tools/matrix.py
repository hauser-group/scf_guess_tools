from __future__ import annotations

from .core import Object
from abc import ABC, abstractmethod
from numpy.typing import Any, NDArray


class Matrix(Object, ABC):
    """Provides a common interface for matrix representations across different
    backends."""

    @property
    @abstractmethod
    def native(self) -> Any:
        """The underlying backend-specific matrix object."""
        pass

    @property
    @abstractmethod
    def size(self) -> float:
        """Total number of elements in the matrix."""
        pass

    @property
    @abstractmethod
    def trace(self) -> float:
        """Trace of the matrix."""
        pass

    @property
    @abstractmethod
    def sum_of_squares(self) -> float:
        """Sum of squares of all matrix elements."""
        pass

    @property
    @abstractmethod
    def numpy(self) -> NDArray:
        """Representation as NumPy arrays. May consist of multiple arrays if symmetry
        is enabled."""
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the matrix."""
        pass

    @abstractmethod
    def __add__(self, other: Matrix) -> Matrix:
        """Matrix addition."""
        pass

    @abstractmethod
    def __sub__(self, other: Matrix) -> Matrix:
        """Matrix subtraction."""
        pass

    @abstractmethod
    def __matmul__(self, other: Matrix) -> Matrix:
        """Matrix multiplication."""
        pass

    @classmethod
    @abstractmethod
    def build(cls, array: NDArray) -> Matrix:
        """Constructs a matrix from a NumPy array."""
        pass
