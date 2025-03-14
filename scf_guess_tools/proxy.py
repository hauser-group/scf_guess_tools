from .core import Backend, cache_directory
from .molecule import Molecule
from .wavefunction import Wavefunction
from joblib import Memory
from time import process_time

_memory = None


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
        global _memory

        if _memory is None:
            if not cache_directory:
                raise RuntimeError("SGT_CACHE environment variable not set")

            _memory = Memory(f"{cache_directory}")

        operation = _memory.cache(operation)

    start = process_time()
    result = operation(*args, **kwargs)
    duration = process_time() - start

    return (result, duration) if time else result


def clear_cache():
    global _memory

    if _memory is not None:
        _memory.clear()


def load(path: str, backend: Backend, cache=False, time=False, **kwargs) -> Molecule:
    return _forward(backend, lambda p: p.Molecule.load, cache, time, path, **kwargs)


def guess(
    molecule: Molecule, basis: str, scheme: str, cache=False, time=False, **kwargs
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
