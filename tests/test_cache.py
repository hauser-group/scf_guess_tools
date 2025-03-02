from common import replace_random_digit
from scf_guess_tools import Engine, PySCFEngine, Psi4Engine

import os
import pytest
import shutil


@pytest.fixture(params=[PySCFEngine, Psi4Engine])
def engine(request, tmp_path, monkeypatch):
    print("creating engine")
    monkeypatch.setenv("PSI_SCRATCH", f"{tmp_path}/psi4")
    monkeypatch.setenv("PYSCF_TMPDIR", f"{tmp_path}/pyscf")
    monkeypatch.setenv("SGT_CACHE", f"{tmp_path}/sgt")

    engine = request.param(cache=True, verbose=1)
    engine.memory.clear()

    return engine


@pytest.fixture(params=["acetaldehyde.xyz", "ch.xyz", "ch2-trip.xyz", "hoclo.xyz"])
def path(request, tmp_path):
    destination = str(tmp_path / request.param)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(request.param, destination)

    return destination


def test_molecule(engine: Engine, path: str):
    molecule = engine.load(path)
    invocations = 0

    @engine.memory.cache
    def function(m):
        nonlocal invocations
        invocations += 1

        return m

    assert function(molecule) == molecule  # not cached
    assert invocations == 1, "properties of molecule must not change"

    assert function(molecule) == molecule  # cached
    assert invocations == 1, "properties of molecule must not change"

    modified_path = path.removesuffix(".xyz") + "-modified.xyz"
    replace_random_digit(path, modified_path)
    modified_molecule = engine.load(modified_path)

    assert function(modified_molecule) == modified_molecule  # not cached
    assert invocations == 2
