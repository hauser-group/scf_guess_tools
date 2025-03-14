from __future__ import annotations

from .core import Object
from ..matrix import Matrix as Base
from numpy.typing import NDArray

import numpy as np


class Matrix(Base, Object):
    @property
    def native(self) -> NDArray:
        return self._native

    @property
    def size(self) -> int:
        return self.native.size

    @property
    def trace(self) -> float:
        return np.trace(self.native)

    @property
    def sum_of_squares(self) -> float:
        return np.sum(self.native**2)

    @property
    def numpy(self) -> NDArray:
        return self._native

    def __init__(self, native: NDArray):
        self._native = native

    def __repr__(self) -> str:
        return self.native.__repr__()

    def __add__(self, other: Matrix) -> Matrix:
        return Matrix(self.native.__add__(other.native))

    def __sub__(self, other: Matrix) -> Matrix:
        return Matrix(self.native.__sub__(other.native))

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix(self.native.__matmul__(other.native))

    def __getstate__(self):
        return super().__getstate__(), self._native

    def __setstate__(self, serialized):
        super().__setstate__(serialized[0])
        self._native = serialized[1]
