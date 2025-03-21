from __future__ import annotations

from provider import context, backend, basis_fixture, path_fixture
from scf_guess_tools import (
    Backend,
    load,
    guess,
    calculate,
    cache,
    clear_cache,
    Molecule,
    Wavefunction,
    Matrix,
)
from scf_guess_tools.common import tuplify
from types import ModuleType
from typing import Callable
import warnings

import pytest

basis = basis_fixture(["sto-3g", "pcseg-0"])
basics_path = path_fixture(max_atoms=15)

wavefunction_path = path_fixture(max_atoms=7)
times_path = path_fixture(charged=False, non_singlet=False, min_atoms=10, max_atoms=20)


@pytest.mark.parametrize("time", [None, True, False])
def test_load(context, backend: Backend, basics_path: str, time: bool | None):
    result = load(basics_path, backend, time=time)

    if time is None or time == False:
        assert isinstance(result, Molecule), "non-timed load must return Molecule"
    else:
        molecule, time = result
        assert isinstance(molecule, Molecule), "timed load must return Molecule"
        assert time > 0.0, "timed load must return positive time"


@pytest.mark.parametrize(
    "time, build, method",
    [
        (time, build, method)
        for time in [None, True, False]
        for build in [guess, calculate]
        for method in ["hf", "dft"]
    ],
)
def test_build(
    context,
    backend: Backend,
    basics_path: str,
    time: bool | None,
    basis: str,
    method: str,
    build: Callable,
):
    molecule = load(basics_path, backend)
    result = build(molecule, basis, time=time, method=method)

    if time is None or time == False:
        assert isinstance(
            result, Wavefunction
        ), "non-timed build function must return Wavefunction"
    else:
        wavefunction, time = result

        assert isinstance(
            wavefunction, Wavefunction
        ), "timed build must return Wavefunction"

        assert time > 0.0, "timed build must return positive time"


@pytest.mark.parametrize(
    "time, attribute, build, method",
    [
        (time, attribute, build, method)
        for time in [None, True, False]
        for attribute in ["overlap", "core_hamiltonian", "density", "fock"]
        for build in [guess, calculate]
        for method in ["hf", "dft"]
    ],
)
def test_wavefunction(
    context,
    backend: Backend,
    wavefunction_path: str,
    time: bool | None,
    basis: str,
    attribute: str,
    method: str,
    build: Callable,
):
    molecule = load(wavefunction_path, backend)
    wavefunction = build(molecule, basis, method=method)

    if attribute == "fock" and method == "dft":
        warnings.warn("Skipping fock matrix for dft")
        return

    method = getattr(wavefunction, attribute)
    result = method(time=time)

    if time is None or time == False:
        assert isinstance(
            tuplify(result)[0], Matrix
        ), "non-timed attribute must return Matrix"
    else:
        matrices, time = result

        assert isinstance(
            tuplify(matrices)[0], Matrix
        ), "timed attribute must return Matrix"

        assert time > 0.0, "timed attribute must return positive time"


@pytest.mark.parametrize("method", ["hf", "dft"])
def test_times(context, backend: Backend, times_path: str, basis: str, method: str):
    molecule = load(times_path, backend)
    initial_1, initial_time_1 = guess(
        molecule, basis, cache=True, time=True, method=method
    )
    final_1, final_time_1 = calculate(
        molecule, basis, cache=True, time=True, method=method
    )

    assert (
        initial_time_1 >= initial_1.time
    ), "invocation time must be bigger than non-cached guessing time"

    assert (
        final_time_1 >= final_1.time
    ), "invocation time must be bigger than non-cached calculation time"

    assert final_time_1 >= initial_time_1, "calculation must take longer than guessing"
    assert final_1.time >= initial_1.time, "calculation must take longer than guessing"

    initial_2, initial_time_2 = guess(
        molecule, basis, cache=True, time=True, method=method
    )
    final_2, final_time_2 = calculate(
        molecule, basis, cache=True, time=True, method=method
    )

    assert initial_2.time == initial_1.time, "cached time must remain unchanged"
    assert final_2.time == final_1.time, "cached time must remain unchanged"

    assert initial_time_2 <= initial_2.time, "loading from cache must be quicker"
    assert final_time_2 <= final_2.time, "loading from cache must be quicker"
