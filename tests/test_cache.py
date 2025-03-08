from common import replace_random_digit
from scf_guess_tools import Engine, PyEngine, PsiEngine

import os
import pytest
import shutil


@pytest.fixture(params=[PyEngine, PsiEngine])
def engine(request, tmp_path, monkeypatch):
    variables = ["PSI_SCRATCH", "PYSCF_TMPDIR", "SGT_CACHE"]
    directories = ["psi4", "pyscf", "sgt"]

    for variable, directory in zip(variables, directories):
        path = f"{tmp_path}/{directory}"
        os.makedirs(path, exist_ok=True)
        monkeypatch.setenv(variable, path)

    engine = request.param(cache=True, verbose=1)
    engine.memory.clear()

    return engine


@pytest.fixture(params=["acetaldehyde.xyz", "ch2-trip.xyz", "hoclo.xyz", "CuMe.xyz"])
def path(request, tmp_path):
    destination = str(tmp_path / request.param)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(request.param, destination)

    return destination


@pytest.fixture(params=["sto-3g"])  # , "pcseg-0", "pcseg-1"])
def basis(request):
    return request.param


def test_molecule(engine: Engine, path: str):
    molecule = engine.load(path)
    invocations = 0

    @engine.memory.cache
    def f(m):
        nonlocal invocations
        invocations += 1

        return m

    assert f(molecule) == molecule, "properties of molecule must not change"
    assert invocations == 1, "function must be invoked for uncached molecule"

    assert f(molecule) == molecule, "properties of molecule must not change"
    assert invocations == 1, "function must not be invoked for cached molecule"

    modified_path = path[:-4] + "-modified.xyz"
    replace_random_digit(path, modified_path)
    modified_molecule = engine.load(modified_path)

    assert f(modified_molecule) != molecule, "properties of molecule most change"
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
    engine.memory.clear()
    molecule = engine.load(path)
    function = getattr(engine, builder)
    wavefunction = function(molecule, basis, scheme)
    invocations = 0

    @engine.memory.cache
    def f(w):
        nonlocal invocations
        invocations += 1

        return w

    assert f(wavefunction) == wavefunction, "properties of wavefunction must not change"
    assert invocations == 1, "function must be invoked for uncached wavefunction"

    assert f(wavefunction) == wavefunction, "properties of wavefunction must not change"
    assert invocations == 1, "function must not be invoked for cached wavefunction"
