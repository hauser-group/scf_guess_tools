from common import equal, replace_random_digit
from provider import context, engine, path, basis
from scf_guess_tools import Engine, PyEngine, PsiEngine

import pytest


@pytest.mark.parametrize(
    "engine, builder",
    [
        (engine, builder)
        for engine in [PyEngine, PsiEngine]
        for builder in ["guess", "calculate"]
    ],
    indirect=["engine"],
)
def test_reinitialization(engine, path: str, builder: str):
    def build(engine):
        molecule = e.load(path)
        return e.calculate(molecule, "pcseg-0")

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
        for scheme in [None, *engine.guessing_schemes()]
    ],
    indirect=["engine"],
)
def test_wavefunction(engine: Engine, path: str, builder: str, scheme: str, basis: str):
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
