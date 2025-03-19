from __future__ import annotations

from .core import Backend, cache_directory, cache as do_cache
from .matrix import Matrix
from .molecule import Molecule
from .wavefunction import Wavefunction

from numpy.typing import NDArray
from time import process_time


def _forward(backend: Backend, get_operation, cache: bool, time: bool, *args, **kwargs):
    """Execute an operation for a specified backend.

    Args:
        backend: The backend to use.
        get_operation: A function retrieving the operation from the backend module.
        cache: Whether to cache the result.
        time: Whether to return execution time.
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
        raise NotImplementedError()

    operation = get_operation(package)

    if cache:
        operation = do_cache()(operation)

    start = process_time()
    result = operation(*args, **kwargs)
    duration = process_time() - start

    return (result, duration) if time else result


def load(path: str, backend: Backend, cache=False, time=False, **kwargs) -> Molecule:
    """Load a molecule from an xyz file.

    Args:
        path: The path to the xyz file.
        backend: The backend to use.
        cache: Whether to cache the result on disk.
        time: Whether to return execution time as a tuple (result, time).
        **kwargs: Additional arguments forwarded to the backend-specific molecule loader.

    Returns:
        The loaded molecule.
    """
    return _forward(backend, lambda p: p.Molecule.load, cache, time, path, **kwargs)


def build(
    array: NDArray, backend: Backend, cache=False, time=False, **kwargs
) -> Matrix:
    """Build a matrix for a specified backend from a NumPy array.

    Args:
        array: The NumPy array used to construct the matrix.
        backend: The backend to use.
        cache: Whether to cache the result on disk.
        time: Whether to return execution time as a tuple (result, time).
        **kwargs: Additional arguments forwarded to the backend's Matrix.build().

    Returns:
        The constructed matrix.
    """
    return _forward(backend, lambda p: p.Matrix.build, cache, time, array, **kwargs)


def guess(
    molecule: Molecule,
    basis: str,
    scheme: str | None = None,
    cache=False,
    time=False,
    **kwargs,
) -> Wavefunction:
    """Create an initial wavefunction guess.

    Args:
        molecule: The molecule for which the wavefunction is created.
        basis: The basis set.
        scheme: The initial guess scheme. Can be one of guessing_schemes(backend). If
            None, the backend's default guessing scheme is used.
        cache: Whether to cache the result on disk.
        time: Whether to return execution time as a tuple (result, time).
        **kwargs: Additional arguments forwarded to the backends Wavefunction.guess().

    Returns:
        The generated wavefunction.
    """
    return _forward(
        molecule.backend(),
        lambda p: p.Wavefunction.guess,
        cache,
        time,
        molecule,
        basis,
        scheme,
        **kwargs,
    )


def calculate(
    molecule: Molecule,
    basis: str,
    initial: str | Wavefunction | None = None,
    method: str = "hf",
    functional: str | None = None,
    cache=False,
    time=False,
    **kwargs,
) -> Wavefunction:
    """Compute a converged wavefunction for the given molecule.

    Args:
        molecule: The molecule for which the wavefunction is computed.
        basis: The basis set.
        initial: The initial guess. Can be one of guessing_schemes(backend) or another
            Wavefunction instance. If None, the backend's default guessing scheme is used.
        method: The calculation method to use (hf, dft)
        functional: The functional to use for dft calculations
        cache: Whether to cache the result on disk.
        time: Whether to return execution time as a tuple (result, time).
        **kwargs: Additional arguments forwarded to the backend's Wavefunction.calculate()

    Returns:
        The computed wavefunction.
    """
    return _forward(
        molecule.backend(),
        lambda p: p.Wavefunction.calculate,
        cache,
        time,
        molecule,
        basis,
        method,
        initial,
        functional,
        **kwargs,
    )
