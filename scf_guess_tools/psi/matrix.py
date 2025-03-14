from __future__ import annotations

from ..matrix import Matrix as Base
from .core import Object
from numpy.typing import NDArray
from psi4.core import Matrix as Native, doublet


class Matrix(Base, Object):
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

    def __getstate__(self):
        raise NotImplementedError()

    def __setstate__(self, serialized):
        raise NotImplementedError()

    def __repr__(self) -> str:
        return "\n".join([irrep.__repr__() for irrep in self.native.nph])

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
