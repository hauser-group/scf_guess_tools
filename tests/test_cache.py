from __future__ import annotations

from common import equal, replace_random_digit
from provider import context, engine, basis_fixture, path_fixture
from scf_guess_tools import (
    Engine,
    PyEngine,
    PsiEngine,
    Metric,
    FScore,
    DIISError,
    EnergyError,
)

import pytest

basis = basis_fixture(["sto-3g", "pcseg-0"])
path = path_fixture(paths=3, medium=False, large=False)


@pytest.mark.parametrize(
    "engine, builder",
    [
        (engine, builder)
        for engine in [PyEngine, PsiEngine]
        for builder in ["guess", "calculate"]
    ],
    indirect=["engine"],
)
def test_reinitialization(engine, path: str, basis: str, builder: str):
    def build(engine):
        molecule = e.load(path)
        return e.calculate(molecule, basis)

    engine = engine.__class__

    e = engine(cache=False, verbose=1)
    unwrapped = getattr(e, builder)
    original = build(e)

    e = engine(cache=True, verbose=1)
    assert getattr(e, builder) != unwrapped, "builder must change"
    assert equal(build(e), original, ignore=["time"]), "output must not change"

    e = engine(cache=False, verbose=1)
    assert getattr(e, builder) == unwrapped, "builder must be restored"
    assert equal(build(e), original, ignore=["time"]), "output must not change"


def test_molecule(engine: Engine, path: str):
    molecule = engine.load(path)
    invocations = 0

    @engine.memory.cache
    def f(m):
        nonlocal invocations
        invocations += 1

        return m

    assert equal(f(molecule), molecule), "properties of molecule must not change"
    assert invocations == 1, "function must be invoked for uncached molecule"

    assert equal(f(molecule), molecule), "properties of molecule must not change"
    assert invocations == 1, "function must not be invoked for cached molecule"

    modified_path = path[:-4] + "-modified.xyz"
    replace_random_digit(path, modified_path)
    modified_molecule = engine.load(modified_path)

    assert not equal(
        f(modified_molecule), molecule
    ), "properties of molecule must change"

    assert invocations == 2, "function must be invoked for uncached molecule"


@pytest.mark.parametrize(
    "engine, builder, scheme",
    [
        (engine, builder, scheme)
        for engine in [PyEngine, PsiEngine]
        for builder in ["guess", "calculate"]
        for scheme in [None, *engine.guessing_schemes()][::2]
    ],
    indirect=["engine"],
)
def test_wavefunction(engine: Engine, path: str, basis: str, scheme: str, builder: str):
    molecule = engine.load(path)
    function = getattr(engine, builder)
    wavefunction = function(molecule, basis, scheme)
    invocations = 0

    @engine.memory.cache
    def f(w):
        nonlocal invocations
        invocations += 1

        return w

    assert equal(
        f(wavefunction), wavefunction
    ), "properties of wavefunction must not change"

    assert invocations == 1, "function must be invoked for uncached wavefunction"

    assert equal(
        f(wavefunction), wavefunction
    ), "properties of wavefunction must not change"

    assert invocations == 1, "function must not be invoked for cached wavefunction"


@pytest.mark.parametrize("builder", ["guess", "calculate"])
def test_time(engine: Engine, path: str, builder: str):
    molecule = engine.load(path)
    function = getattr(engine, builder)
    wavefunction = function(molecule, "pcseg-0")

    assert (
        wavefunction.time <= wavefunction.load_time
    ), "cached invocation must take longer than actual wavefunction calculation"

    engine = engine.__class__(cache=False, verbose=1)
    molecule = engine.load(path)
    function = getattr(engine, builder)
    wavefunction = function(molecule, "pcseg-0")

    assert wavefunction.time > 0.0, "wavefunction building must take some time"

    assert (
        wavefunction.load_time is None
    ), "non-cached wavefunction building must not have loading time"


@pytest.mark.parametrize("metric", [FScore, EnergyError])
def test_metric(engine: Engine, path: str, metric: Metric):
    molecule = engine.load(path)
    initial = engine.guess(molecule, "pcseg-0")
    final = engine.calculate(molecule, "pcseg-0")

    m = metric(initial)

    assert m.final.time == final.time, "cached wavefunction must have same build time"
    assert m.final.load_time is not None, "cached wavefunction must have load time"
