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

    def __eq__(self, other: Matrix) -> bool:
        return np.array_equal(self.native, other.native)
