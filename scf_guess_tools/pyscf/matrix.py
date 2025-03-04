from __future__ import annotations

from ..matrix import Matrix as Base
from numpy.typing import NDArray

import numpy as np


class Matrix(Base):
    def __init__(self, native: NDArray):
        self._native = native

    @property
    def native(self) -> NDArray:
        return self._native

    def __repr__(self) -> str:
        return self.native.__repr__()

    def __eq__(self, other: Matrix) -> bool:
        return np.array_equal(self.native, other.native)

    def __add__(self, other: Matrix) -> Matrix:
        return Matrix(self.native.__add__(other.native))

    def __sub__(self, other: Matrix) -> Matrix:
        return Matrix(self.native.__sub__(other.native))

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix(self.native.__matmul__(other.native))

    @property
    def size(self) -> int:
        return self.native.size

    @property
    def trace(self) -> float:
        return np.trace(self.native)

    @property
    def sum_of_squares(self) -> float:
        return np.sum(self.native**2)
