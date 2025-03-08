from __future__ import annotations

from ..matrix import Matrix as Base
from psi4.core import Matrix as Native, doublet
from numpy.typing import NDArray

import numpy as np


class Matrix(Base):
    @property
    def native(self) -> Native:
        return self._native

    @property
    def size(self) -> int:
        shape = (self.native.shape,) if self.native.nirrep() == 1 else self.native.shape
        rows = sum([i for i, _ in shape])
        columns = sum([j for _, j in shape])
        return rows * columns

    @property
    def trace(self) -> float:
        return self.native.trace()

    @property
    def sum_of_squares(self) -> float:
        return self.native.sum_of_squares()

    @property
    def numpy(self) -> NDArray:
        return self.native.to_array(dense=True)

    def __init__(self, native: Native):
        self._native = native

    def __repr__(self) -> str:
        return "\n".join([irrep.__repr__() for irrep in self.native.nph])

    def __eq__(self, other: Matrix) -> bool:
        for a, b in zip(self.native.nph, other.native.nph):
            if not np.allclose(a, b, rtol=1e-5, atol=1e-10):
                return False

        return True

    def __add__(self, other: Matrix) -> Matrix:
        result = self.native.clone()
        result.add(other.native)

        return Matrix(result)

    def __sub__(self, other: Matrix) -> Matrix:
        result = self.native.clone()
        result.subtract(other.native)

        return Matrix(result)

    def __matmul__(self, other: Matrix) -> Matrix:
        return Matrix(doublet(self.native, other.native))
