from __future__ import annotations

from .core import Object
from ..matrix import Matrix as Base
from numpy.typing import NDArray

import numpy as np


class Matrix(Base, Object):
    """Matrix representation using the PySCF backend. This class provides an
    implementation of the Matrix interface using NumPy arrays as the underlying
    representation.
    """

    @property
    def native(self) -> NDArray:
        """The underlying NumPy array."""
        return self._native

    @property
    def size(self) -> int:
        """Total number of elements in the matrix."""
        return self.native.size

    @property
    def trace(self) -> float:
        """Trace of the matrix."""
        return np.trace(self.native)

    @property
    def sum_of_squares(self) -> float:
        """Sum of squares of all matrix elements."""
        return np.sum(self.native**2)

    @property
    def numpy(self) -> NDArray:
        """NumPy array representation of the matrix."""
        return self._native

    def __init__(self, native: NDArray):
        """Initialize the PySCF matrix.

        Args:
            native: The NumPy array to wrap.
        """
        self._native = native

    def __repr__(self) -> str:
        """String representation of the matrix."""
        return self.native.__repr__()

    def __getstate__(self):
        """Serialize the matrix for pickling."""
        return super().__getstate__(), self._native

    def __setstate__(self, serialized):
        """Restore the matrix from a serialized state.

        Args:
            serialized: The serialized matrix data.
        """
        super().__setstate__(serialized[0])
        self._native = serialized[1]

    def __add__(self, other: Matrix) -> Matrix:
        """Matrix addition.

        Args:
            other: The matrix to add.

        Returns:
            A new Matrix instance representing the sum.
        """
        return Matrix(self.native.__add__(other.native))

    def __sub__(self, other: Matrix) -> Matrix:
        """Matrix subtraction.

        Args:
            other: The matrix to subtract.

        Returns:
            A new Matrix instance representing the difference.
        """
        return Matrix(self.native.__sub__(other.native))

    def __matmul__(self, other: Matrix) -> Matrix:
        """Matrix multiplication.

        Args:
            other: The matrix to multiply with.

        Returns:
            A new Matrix instance representing the product.
        """
        return Matrix(self.native.__matmul__(other.native))

    @classmethod
    def build(cls, array: NDArray) -> Matrix:
        """Construct a PySCF matrix from a NumPy array.

        Args:
            array: The NumPy array used to initialize the matrix.

        Returns:
            A Matrix instance wrapping the NumPy array.
        """
        return Matrix(array)
