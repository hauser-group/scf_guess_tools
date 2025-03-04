from __future__ import annotations

from ..common import tuplify
from ..matrix import Matrix as Base
from psi4.core import Matrix as Native, doublet

import numpy as np


class Matrix(Base):
    def __init__(self, native: Native = None, npp=None):
        self._native = native

        if npp is not None:
            self._np = npp
        else:
            self._np = native.to_array(copy=True, dense=True)
            if self._native is None:
                self._native = Native.from_array(self._np)

    @property
    def native(self) -> Native:
        return self._native

    def __repr__(self) -> str:
        return "\n".join([irrep.__repr__() for irrep in self.native.nph])

    def __eq__(self, other: Matrix) -> bool:
        return False
        for a, b in zip(self.native.nph, other.native.nph):
            if not np.array_equal(a, b):
                return False

        return True

    def __add__(self, other: Matrix) -> Matrix:
        assert False
        result = self.native.clone()
        result.add(other.native)

        return Matrix(npp=result)

    def __sub__(self, other: Matrix) -> Matrix:
        # result = self.native.clone()
        # result.subtract(other.native)

        return Matrix(npp=self._np - other._np)

    def __matmul__(self, other: Matrix) -> Matrix:
        res = self._np @ other._np
        return Matrix(npp=res)

        # return Matrix(doublet(self.native, other.native))

    @property
    def size(self) -> int:
        return self._np.size

        shape = (self.native.shape,) if self.native.nirrep() == 1 else self.native.shape
        rows = sum([i for i, _ in shape])
        columns = sum([j for _, j in shape])
        return rows * columns

    @property
    def trace(self) -> float:
        return self._np.trace
        # return self.native.trace()

    @property
    def sum_of_squares(self) -> float:
        return np.sum(self._np**2)
        # return self.native.sum_of_squares()
