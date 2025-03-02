from ..matrix import Matrix as Base
from psi4.core import Matrix as Native
from typing import Self

import numpy as np


class Matrix(Base):
    def __init__(self, native: Native):
        self._native = native

    @property
    def native(self) -> Native:
        return self._native

    def __eq__(self, other: Self) -> bool:
        return np.array_equal(self.native.nph, other.native.nph)
