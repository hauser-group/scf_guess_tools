from __future__ import annotations
from pathlib import Path

from common import equal, replace_random_digit, similar, tuplify
from provider import context, backend, basis_fixture, path_fixture
from scf_guess_tools import (
    Backend,
    load,
    guess,
    calculate,
    cache,
    clear_cache,
    f_score,
    diis_error,
    energy_error,
    guessing_schemes,
)
from types import ModuleType
from typing import Callable

import numpy as np
import pytest
import warnings

from molecules.reference import refernces


@pytest.mark.parametrize(
    "name, method, basis",
    [
        (name, method, basis)
        for name, entries in refernces.items()
        for method, basis_dict in entries.items()
        for basis in basis_dict
    ],
)
def test_regression_energy_match(context, name: str, method: str, basis: str):

    backends = [Backend.PSI, Backend.PY]
    tolerance = 1e-1  # Hartree

    path = str(Path(__file__).parent / "molecules" / "geometries" / f"{name}.xyz")

    for backend in backends:
        try:
            molecule = load(path, backend)
            if method == "hf":
                result = calculate(molecule, basis, method=method)
            else:
                result = calculate(molecule, basis, method="dft", functional=method)

            # if not result.converged or not result.stable:  #! do not check for now
            #     warnings.warn(f"{name} ({backend}, {method}) not converged or stable")
            #     continue

            E_sim = result.electronic_energy() + result.nuclear_repulsion_energy()
            E_ref = refernces[name][method][basis]

            assert similar(E_sim, E_ref, tolerance)

        except FileNotFoundError:
            warnings.warn(f"Geometry for {name} not found in backend {backend}")
