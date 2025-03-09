from molecules.properties import properties
from pathlib import Path
from scf_guess_tools import PyEngine, PsiEngine

import os
import pytest
import shutil
import random


@pytest.fixture
def context():
    directories = [
        os.environ.get(variable)
        for variable in ["PSI_SCRATCH", "PYSCF_TMPDIR", "SGT_CACHE"]
    ]

    for directory in directories:
        try:
            shutil.rmtree(f"{directory}.pytest-bak")
        except:
            pass

        try:
            shutil.move(directory, f"{directory}.pytest-bak")
        except:
            pass
        finally:
            os.makedirs(directory, exist_ok=True)

    try:
        yield
    finally:
        for directory in directories:
            try:
                shutil.rmtree(directory)
            except:
                pass

            try:
                shutil.move(f"{directory}.pytest-bak", directory)
            except:
                pass


@pytest.fixture(params=[PyEngine, PsiEngine])
def engine(request, context):
    engine = request.param(cache=True, verbose=1)

    try:
        engine.memory.clear()
    finally:
        return engine


def basis_fixture(bases: list[str]):
    @pytest.fixture(params=bases)
    def basis(request):
        return request.param

    return basis


def path_fixture(
    paths: int,
    charged: bool = None,
    singlet: bool = None,
    small: bool = None,
    medium: bool = None,
    large: bool = None,
):
    matches = []

    for name, attributes in properties.items():
        if charged == True and attributes["charge"] == 0:
            continue

        if charged == False and attributes["charge"] != 0:
            continue

        if singlet == True and attributes["multiplicity"] != 1:
            continue

        if singlet == False and attributes["multiplicity"] == 1:
            continue

        if small == True and attributes["atoms"] >= 10:
            continue

        if small == False and attributes["atoms"] < 10:
            continue

        if medium == True and (attributes["atoms"] < 10 or attributes["atoms"] > 30):
            continue

        if medium == False and (
            attributes["atoms"] >= 10 and attributes["atoms"] <= 30
        ):
            continue

        if large == True and attributes["atoms"] <= 30:
            continue

        if large == False and attributes["atoms"] > 30:
            continue

        matches.append(name)

    random.seed(42)
    matches = random.sample(matches, min(paths, len(matches)))

    @pytest.fixture(params=matches)
    def path(request, tmp_path):
        source = (
            Path(__file__).parent / "molecules" / "geometries" / f"{request.param}.xyz"
        )
        destination = f"{tmp_path}/{request.param}.xyz"

        shutil.copyfile(source, destination)
        return destination

    return path
