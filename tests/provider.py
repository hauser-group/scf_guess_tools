from __future__ import annotations

from collections import defaultdict
from molecules.properties import properties
from pathlib import Path
from scf_guess_tools import Backend, reset
from scf_guess_tools.psi import reset as reset_psi
from scf_guess_tools.py import reset as reset_py

import os
import pytest
import random
import shutil


@pytest.fixture
def context():
    variables = ["PSI_SCRATCH", "PYSCF_TMPDIR", "SGT_CACHE"]
    directories = [os.environ.get(variable) for variable in variables]

    for directory, variable in zip(directories, variables):
        if directory is None:
            pytest.fail(f"{variable} environment variable not set")

        try:
            shutil.rmtree(directory)
        except:
            pass
        finally:
            os.makedirs(directory, exist_ok=True)

    reset()
    reset_psi()
    reset_py()

    try:
        yield
    finally:
        for directory in directories:
            shutil.rmtree(directory)


@pytest.fixture(params=[Backend.PSI, Backend.PY])
def backend(request):
    return request.param


def basis_fixture(bases: list[str]):
    @pytest.fixture(params=bases)
    def basis(request):
        return request.param

    return basis


def path_fixture(
    charged: int = 1,
    non_charged: int = 1,
    singlet: int = 1,
    non_singlet: int = 1,
    min_atoms: int = 0,
    max_atoms: int = 1000,
):
    mapping = {
        "chargedTrue": charged,
        "chargedFalse": non_charged,
        "singletTrue": singlet,
        "singletFalse": non_singlet,
    }

    categories = defaultdict(set)
    atoms = {}

    for name, attributes in properties.items():
        if attributes["atoms"] < min_atoms or attributes["atoms"] > max_atoms:
            continue

        categories[f"charged{attributes['charge'] != 0}"].add(name)
        categories[f"singlet{attributes['multiplicity'] == 1}"].add(name)

    random.seed(42)
    for category, names in categories.items():
        unique = sorted(list(names))
        random.shuffle(unique)

        categories[category] = unique

    from constraint import Problem, AllDifferentConstraint

    problem = Problem()
    variables = []

    for category, number in mapping.items():
        for i in range(number):
            variable = f"{category}-{i}"
            problem.addVariable(variable, list(categories[category]))
            variables.append(variable)

    problem.addConstraint(AllDifferentConstraint(), variables)
    solution = problem.getSolution()

    assert solution is not None

    @pytest.fixture(params=solution.values())
    def path(request, tmp_path):
        source = (
            Path(__file__).parent / "molecules" / "geometries" / f"{request.param}.xyz"
        )
        destination = f"{tmp_path}/{request.param}.xyz"

        shutil.copyfile(source, destination)
        return destination

    return path
