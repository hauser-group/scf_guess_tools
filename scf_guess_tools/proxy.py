from __future__ import annotations

from .core import Backend
from .matrix import Matrix
from .molecule import Molecule
from .wavefunction import Wavefunction

from numpy.typing import NDArray


def _forward(backend: Backend, get_operation, *args, **kwargs):
    """Execute an operation for a specified backend.

    Args:
        backend: The backend to use.
        get_operation: A function retrieving the operation from the backend module.
        *args: Positional arguments passed to the operation.
        **kwargs: Keyword arguments passed to the operation.

    Returns:
        The operation result, optionally with execution time.
    """
    package = None

    if backend == Backend.PSI:
        from . import psi

        package = psi
    elif backend == Backend.PY:
        from . import py

        package = py
    else:
        raise ValueError(f"Invalid backend {backend}")

    operation = get_operation(package)
    return operation(*args, **kwargs)


def load(path: str, backend: Backend, **kwargs) -> Molecule:
    """Load a molecule from an xyz file.

    Args:
        path: The path to the xyz file.
        backend: The backend to use.
        **kwargs: Additional arguments forwarded to the backend-specific molecule loader.

    Returns:
        The loaded molecule.
    """
    return _forward(backend, lambda p: p.Molecule.load, path, **kwargs)


def build(array: NDArray, backend: Backend, **kwargs) -> Matrix:
    """Build a matrix for a specified backend from a NumPy array.

    Args:
        array: The NumPy array used to construct the matrix.
        backend: The backend to use.
        **kwargs: Additional arguments forwarded to the backend's Matrix.build().

    Returns:
        The constructed matrix.
    """
    return _forward(backend, lambda p: p.Matrix.build, array, **kwargs)


def guess(
    molecule: Molecule,
    basis: str,
    scheme: str | None = None,
    **kwargs,
) -> Wavefunction:
    """Create an initial wavefunction guess.

    Args:
        molecule: The molecule for which the wavefunction is created.
        basis: The basis set.
        scheme: The initial guess scheme. Can be one of guessing_schemes(backend). If
            None, the backend's default guessing scheme is used.
        **kwargs: Additional arguments forwarded to the backends Wavefunction.guess().

    Returns:
        The generated wavefunction.
    """
    return _forward(
        molecule.backend(),
        lambda p: p.Wavefunction.guess,
        molecule,
        basis,
        scheme,
        **kwargs,
    )


def calculate(
    molecule: Molecule,
    basis: str,
    initial: str | Wavefunction | None = None,
    **kwargs,
) -> Wavefunction:
    """Compute a converged wavefunction for the given molecule.

    Args:
        molecule: The molecule for which the wavefunction is computed.
        basis: The basis set.
        initial: The initial guess. Can be one of guessing_schemes(backend) or another
            Wavefunction instance. If None, the backend's default guessing scheme is used.
        **kwargs: Additional arguments forwarded to the backend's Wavefunction.calculate()

    Returns:
        The computed wavefunction.
    """
    return _forward(
        molecule.backend(),
        lambda p: p.Wavefunction.calculate,
        molecule,
        basis,
        initial,
        **kwargs,
    )
