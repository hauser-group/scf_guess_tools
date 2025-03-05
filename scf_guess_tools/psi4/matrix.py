from __future__ import annotations

from ..common import tuplify
from ..matrix import Matrix as Base
from psi4.core import Matrix as Native, doublet

import numpy as np


class Matrix(Base):
    def __init__(self, native: Native):
        self._native = native

    @property
    def native(self) -> Native:
        return self._native

    def __repr__(self) -> str:
        return "\n".join([irrep.__repr__() for irrep in self.native.nph])

    def __eq__(self, other: Matrix) -> bool:
        for a, b in zip(self.native.nph, other.native.nph):
            if not np.array_equal(a, b):
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
