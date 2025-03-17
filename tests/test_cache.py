from __future__ import annotations

from common import equal, replace_random_digit
from provider import context, backend, basis_fixture, path_fixture
from scf_guess_tools import Backend, psi, py, load, guess, calculate, cache, clear_cache
from types import ModuleType
from typing import Callable

import pytest

basis = basis_fixture(["sto-3g", "pcseg-0"])
caching_path = path_fixture()
molecule_path = path_fixture()
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
    "backend_package, builder, scheme, symmetry",
    [
        (backend_package, builder, scheme, symmetry)
        for backend_package in zip([Backend.PSI, Backend.PY], [psi, py])
        for builder in [guess, calculate]
        for scheme in [None] + backend_package[1].guessing_schemes
        for symmetry in [True, False]
    ],
)
def test_wavefunction(
    context,
    backend_package: tuple[Backend, ModuleType],
    wavefunction_path: str,
    basis: str,
    scheme: str,
    builder: Callable,
    symmetry: bool,
):
    backend, package = backend_package
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
