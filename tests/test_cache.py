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
mixed_path = path_fixture(max_atoms=7)


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
    original = load(molecule_path, backend, symmetry=symmetry)
    invocations = 0

    @cache()
    def f(m):
        nonlocal invocations
        invocations += 1

        return m

    uncached = f(original)

    assert equal(uncached, original), "properties of molecule must not change"
    assert hash(uncached) == hash(original), "hash of molecule must not change"
    assert invocations == 1, "function must be invoked for non-cached molecule"

    cached = f(original)

    assert equal(cached, original), "properties of molecule must not change"
    assert hash(cached) == hash(original), "hash of molecule must not change"
    assert invocations == 1, "function must not be invoked for cached molecule"

    modified_path = molecule_path[:-4] + "-modified.xyz"
    replace_random_digit(molecule_path, modified_path)
    modified_molecule = load(modified_path, backend)

    modified = f(modified_molecule)

    assert not equal(modified, original), "properties of molecule must change"
    assert invocations == 2, "function must be invoked for non-cached molecule"


@pytest.mark.parametrize(
    "backend, builder, method, scheme, symmetry",
    [
        (backend, builder, method, scheme, symmetry)
        for backend in [Backend.PSI, Backend.PY]
        for builder in [guess, calculate]
        for method in ["hf", "dft"]
        for scheme in [None, *guessing_schemes(backend)][::2]
        for symmetry in [True, False]
    ],
)
def test_matrix(
    context,
    backend: Backend,
    matrix_path: str,
    basis: str,
    builder: Callable,
    method: str,
    scheme: str,
    symmetry: bool,
):
    molecule = load(matrix_path, backend, symmetry=symmetry)
    if builder == guess:
        wavefunction = guess(molecule, basis, scheme, method=method)
    else:
        wavefunction = calculate(molecule, basis, scheme, method=method)

    matrices = [wavefunction.overlap(), wavefunction.core_hamiltonian()]
    matrices.extend(wavefunction.density(tuplify=True))
    if method == "hf":
        matrices.extend(wavefunction.fock(tuplify=True))

    for original in matrices:
        clear_cache()
        invocations = 0

        @cache()
        def f(m):
            nonlocal invocations
            invocations += 1

            return m

        uncached = f(original)

        assert equal(uncached, original), "properties of matrix must not change"
        assert hash(uncached) == hash(original), "hash of matrix must not change"
        assert invocations == 1, "function must be invoked for non-cached matrix"

        cached = f(original)

        assert equal(cached, original), "properties of matrix must not change"
        assert hash(cached) == hash(original), "hash of matrix must not change"
        assert invocations == 1, "function must not be invoked for cached matrix"

        clone = build(original.numpy, backend)
        assert equal(clone, original), "properties of clone must not change"
        assert hash(clone) == hash(original), "hash of clone must not change"


@pytest.mark.parametrize(
    "backend, builder, method, scheme, symmetry",
    [
        (backend, builder, method, scheme, symmetry)
        for backend in [Backend.PSI, Backend.PY]
        for builder in [guess, calculate]
        for method in ["hf", "dft"]
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
    method: str,
    builder: Callable,
    symmetry: bool,
):
    molecule = load(wavefunction_path, backend, symmetry=symmetry)
    if builder == guess:
        original = guess(molecule, basis, scheme, method=method)
    else:
        original = calculate(molecule, basis, scheme, method=method)
    invocations = 0

    @cache()
    def f(w):
        nonlocal invocations
        invocations += 1

        return w

    uncached = f(original)

    to_ignore = []
    if method == "dft":  # dft doesn't give fock matrix!
        to_ignore = ["fock", "electronic_energy"]

    assert equal(
        uncached, original, ignore=to_ignore
    ), "properties of wavefunction must not change"
    assert hash(uncached) == hash(original), "hash of wavefunction must not change"
    assert invocations == 1, "function must be invoked for non-cached wavefunction"

    cached = f(original)

    assert equal(
        cached, original, ignore=to_ignore
    ), "properties of wavefunction must not change"
    assert hash(cached) == hash(original), "hash of wavefunction must not change"
    assert invocations == 1, "function must not be invoked for cached wavefunction"


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
def test_mixed(
    context,
    backend: Backend,
    mixed_path: str,
    scheme: str,
    builder: Callable,
    method: str,
    symmetry: bool,
):
    basis = "pcseg-0"
    c_invocations, g_invocations = 0, 0
    to_ignore = []
    if method == "dft":  # dft doesn't give fock matrix!
        to_ignore = ["fock", "electronic_energy"]

    @cache()
    def c(*args, **kwargs):
        nonlocal c_invocations
        c_invocations += 1
        return calculate(*args, **kwargs)

    @cache()
    def g(*args, **kwargs):
        nonlocal g_invocations
        g_invocations += 1
        return guess(*args, **kwargs)

    molecule1 = load(mixed_path, backend, symmetry=symmetry)
    molecule2 = load(mixed_path, backend, symmetry=symmetry)
    assert hash(molecule1) == hash(molecule2), "molecules must be identical"

    initial1 = g(molecule1, basis, scheme)
    assert g_invocations == 1, "guess must be invoked once"
    final1 = c(molecule1, basis, initial1)
    assert c_invocations == 1, "calculate must be invoked once"

    initial2 = g(molecule2, basis, scheme)
    assert g_invocations == 1, "guess should be cached now"
    final2 = c(molecule2, basis, initial2)
    assert c_invocations == 1, "calculate must be invoked once"

    f_invocations = 0

    @cache()
    def f(w):
        nonlocal f_invocations
        f_invocations += 1

        return w

    uncached = f(final1)

    assert equal(
        uncached, final1, to_ignore
    ), "properties of wavefunction must not change"
    assert hash(uncached) == hash(final1), "hash of wavefunction must not change"
    assert f_invocations == 1, "function must be invoked for non-cached wavefunction"

    cached = f(final2)

    assert equal(
        cached, final1, to_ignore
    ), "properties of wavefunction must not change"
    assert hash(cached) == hash(final1), "hash of wavefunction must not change"
    assert f_invocations == 1, "function must not be invoked for cached wavefunction"
