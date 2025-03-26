from __future__ import annotations

from ..matrix import Matrix as Base
from .core import Object
from numpy.typing import NDArray
from psi4.core import Matrix as Native, doublet


class Matrix(Base, Object):
    """Matrix representation using the Psi4 backend. This class provides an
    implementation of the Matrix interface using Psi4's native matrix format.
    """

    @property
    def native(self) -> Native:
        """The underlying Psi4 matrix object."""
        return self._native

    @property
    def size(self) -> int:
        """Total number of elements in the matrix."""
        shape = (self.native.shape,) if self.native.nirrep() == 1 else self.native.shape
        rows = sum([i for i, _ in shape])
        columns = sum([j for _, j in shape])
        return rows * columns

    @property
    def trace(self) -> float:
        """Trace of the matrix."""
        return self.native.trace()

    @property
    def sum_of_squares(self) -> float:
        """Sum of squares of all matrix elements."""
        return self.native.sum_of_squares()

    @property
    def numpy(self) -> NDArray:
        """NumPy array representation of the matrix."""
        return self.native.to_array(dense=False)

    def __init__(self, native: Native):
        """Initialize the Psi4 matrix.

        Args:
            native: The Psi4 Matrix instance to wrap.
        """
        self._native = native

    def __repr__(self) -> str:
        """String representation of the matrix."""
        return "\n".join([irrep.__repr__() for irrep in self.native.nph])

    def __getstate__(self):
        """Serialize the matrix for pickling."""
        return super().__getstate__(), self._native.to_serial()

    def __setstate__(self, serialized):
        """Restore the matrix from a serialized state.

        Args:
            serialized: The serialized matrix data.
        """
        super().__setstate__(serialized[0])
        self._native = Native.from_serial(serialized[1])

    def __add__(self, other: Matrix) -> Matrix:
        """Matrix addition.

        Args:
            other: The matrix to add.

        Returns:
            A new Matrix instance representing the sum.
        """
        result = self.native.clone()
        result.add(other.native)

        return Matrix(result)

    def __sub__(self, other: Matrix) -> Matrix:
        """Matrix subtraction.

        Args:
            other: The matrix to subtract.

        Returns:
            A new Matrix instance representing the difference.
        """
        result = self.native.clone()
        result.subtract(other.native)

        return Matrix(result)

    def __matmul__(self, other: Matrix) -> Matrix:
        """Matrix multiplication.

        Args:
            other: The matrix to multiply with.

        Returns:
            A new Matrix instance representing the product.
        """
        return Matrix(doublet(self.native, other.native))

    @classmethod
    def build(cls, array: NDArray) -> Matrix:
        """Construct a Psi4 matrix from a NumPy array.

        Args:
            array: The NumPy array used to initialize the matrix.

        Returns:
            A Matrix instance wrapping the Psi4 native matrix.
        """
        return Matrix(Native.from_array(array))
