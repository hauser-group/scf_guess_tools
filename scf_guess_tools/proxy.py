from __future__ import annotations

from .core import Backend, cache_directory, cache as do_cache
from .matrix import Matrix
from .molecule import Molecule
from .wavefunction import Wavefunction

from numpy.typing import NDArray
from time import process_time


def _forward(backend: Backend, get_operation, cache: bool, time: bool, *args, **kwargs):
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
    return _forward(backend, lambda p: p.Molecule.load, cache, time, path, **kwargs)


def build(
    array: NDArray, backend: Backend, cache=False, time=False, **kwargs
) -> Matrix:
    return _forward(backend, lambda p: p.Matrix.build, cache, time, array, **kwargs)


def guess(
    molecule: Molecule,
    basis: str,
    scheme: str | None = None,
    cache=False,
    time=False,
    **kwargs,
) -> Wavefunction:
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
    cache=False,
    time=False,
    **kwargs,
) -> Wavefunction:
    return _forward(
        molecule.backend(),
        lambda p: p.Wavefunction.calculate,
        cache,
        time,
        molecule,
        basis,
        initial,
        **kwargs,
    )
