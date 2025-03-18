from __future__ import annotations

from common import equal, replace_random_digit
from provider import context, backend, basis_fixture, path_fixture
from scf_guess_tools import (
    Backend,
    load,
    build,
    guess,
    calculate,
    cache,
    clear_cache,
    guessing_schemes,
)
from types import ModuleType
from typing import Callable

import pytest

basis = basis_fixture(["sto-3g", "pcseg-0"])
caching_path = path_fixture()
molecule_path = path_fixture()
matrix_path = path_fixture(max_atoms=7)
wavefunction_path = path_fixture(max_atoms=7)


def test_basics(context):
    invocations = 0

    @cache()
    def increment(input):
        nonlocal invocations
        invocations += 1

        return input + 1

    assert invocations == 0, "annotation must not invoke function by itself"
    assert increment(1) == 2, "function must work properly"
    assert invocations == 1, "function must be invoked for non-cached result"
    assert increment(1) == 2, "function must work properly"
    assert invocations == 1, "function must not be invoked for cached result"
    assert increment(2) == 3, "function must work properly"
    assert invocations == 2, "function must be invoked for non-cached result"

    clear_cache()

    assert increment(1) == 2, "function must work properly"
    assert invocations == 3, "result must not be known if cache was cleared"
    assert increment(2) == 3, "function must work properly"
    assert invocations == 4, "result must not be known if cache was cleared"
    assert increment(2) == 3, "function must work properly"
    assert invocations == 4, "function must not be invoked for cached result"


def test_ignore(context):
    invocations = 0

    @cache(ignore=["debug"])
    def increment(input, debug):
        nonlocal invocations
        invocations += 1

        return input + 1

    assert increment(1, True) == 2, "function must work properly"
    assert invocations == 1, "function must be invoked for non-cached result"
    assert increment(1, False) == 2, "function must work properly"
    assert invocations == 1, "function must not be invoked for cached result"


def test_disable(context):
    invocations = 0

    @cache()
    def increment(input):
        nonlocal invocations
        invocations += 1

        return input + 1

    assert increment(1) == 2, "function must work properly"
    assert invocations == 1, "function must be invoked for non-cached result"
    assert increment(1, cache=False) == 2, "function must work properly"
    assert invocations == 2, "function must be invoked for disabled cache"
    assert increment(1, cache=True) == 2, "function must work properly"
    assert invocations == 2, "function must not be invoked for cached result"


@pytest.mark.parametrize("symmetry", [True, False])
def test_molecule(context, backend: Backend, molecule_path: str, symmetry: bool):
    molecule = load(molecule_path, backend, symmetry=symmetry)
    invocations = 0

    @cache()
    def f(m):
        nonlocal invocations
        invocations += 1

        return m

    assert equal(f(molecule), molecule), "properties of molecule must not change"
    assert invocations == 1, "function must be invoked for non-cached molecule"

    assert equal(f(molecule), molecule), "properties of molecule must not change"
    assert invocations == 1, "function must not be invoked for cached molecule"

    modified_path = molecule_path[:-4] + "-modified.xyz"
    replace_random_digit(molecule_path, modified_path)
    modified_molecule = load(modified_path, backend)

    assert not equal(
        f(modified_molecule), molecule
    ), "properties of molecule must change"

    assert invocations == 2, "function must be invoked for non-cached molecule"


@pytest.mark.parametrize(
    "backend, builder, scheme, symmetry",
    [
        (backend, builder, scheme, symmetry)
        for backend in [Backend.PSI, Backend.PY]
        for builder in [guess, calculate]
        for scheme in [None, *guessing_schemes(backend)][::2]
        for symmetry in [True, False]
    ],
)
def test_matrix(
    context,
    backend: Backend,
    matrix_path: str,
    basis: str,
    scheme: str,
    builder: Callable,
    symmetry: bool,
):
    molecule = load(matrix_path, backend, symmetry=symmetry)
    wavefunction = builder(molecule, basis, scheme)

    matrices = [wavefunction.overlap(), wavefunction.core_hamiltonian()]
    matrices.extend(wavefunction.density(tuplify=True))
    matrices.extend(wavefunction.fock(tuplify=True))

    for matrix in matrices:
        clear_cache()
        invocations = 0

        @cache()
        def f(m):
            nonlocal invocations
            invocations += 1

            return m

        assert equal(f(matrix), matrix), "properties of matrix must not change"
        assert invocations == 1, "function must be invoked for non-cached matrix"

        assert equal(f(matrix), matrix), "properties of matrix must not change"
        assert invocations == 1, "function must not be invoked for cached matrix"

        new = build(matrix.numpy, backend)
        assert equal(new, matrix), "properties of matrix must not change"


@pytest.mark.parametrize(
    "backend, builder, scheme, symmetry",
    [
        (backend, builder, scheme, symmetry)
        for backend in [Backend.PSI, Backend.PY]
        for builder in [guess, calculate]
        for scheme in [None, *guessing_schemes(backend)][::2]
        for symmetry in [True, False]
    ],
)
def test_wavefunction(
    context,
    backend: Backend,
    wavefunction_path: str,
    basis: str,
    scheme: str,
    builder: Callable,
    symmetry: bool,
):
    molecule = load(wavefunction_path, backend, symmetry=symmetry)
    wavefunction = builder(molecule, basis, scheme)
    invocations = 0

    @cache()
    def f(w):
        nonlocal invocations
        invocations += 1

        return w

    assert equal(
        f(wavefunction), wavefunction
    ), "properties of wavefunction must not change"

    assert invocations == 1, "function must be invoked for non-cached wavefunction"

    assert equal(
        f(wavefunction), wavefunction
    ), "properties of wavefunction must not change"

    assert invocations == 1, "function must not be invoked for cached wavefunction"
