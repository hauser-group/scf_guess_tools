from scf_guess_tools import PyEngine, PsiEngine

import os
import pytest
import shutil


@pytest.fixture
def context(tmp_path, monkeypatch):
    variables = ["PSI_SCRATCH", "PYSCF_TMPDIR", "SGT_CACHE"]
    directories = ["psi4", "pyscf", "sgt"]

    for variable, directory in zip(variables, directories):
        path = f"{tmp_path}/{directory}"
        os.makedirs(path, exist_ok=True)
        monkeypatch.setenv(variable, path)


@pytest.fixture(params=[PyEngine, PsiEngine])
def engine(request, context):
    engine = request.param(cache=True, verbose=1)
    engine.memory.clear()

    return engine


@pytest.fixture(params=["acetaldehyde.xyz", "ch2-trip.xyz", "hoclo.xyz"])  # "CuMe.xyz"
def path(request, tmp_path):
    destination = str(tmp_path / request.param)

    os.makedirs(os.path.dirname(destination), exist_ok=True)
    shutil.copyfile(request.param, destination)

    return destination


@pytest.fixture(params=["sto-3g"])  # , "pcseg-0", "pcseg-1"])
def basis(request):
    return request.param
