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
    "name, method, basis, backend",
    [
        (name, method, basis, backend)
        for name, entries in refernces.items()
        for method, basis_dict in entries.items()
        for basis in basis_dict
        for backend in [Backend.PSI, Backend.PY]
    ],
)
def test_regression_energy_match(
    context, backend: Backend, name: str, method: str, basis: str
):
    tolerance = 1e-1  # Hartree

    path = str(Path(__file__).parent / "molecules" / "geometries" / f"{name}.xyz")

    try:
        molecule = load(path, backend)
        if method == "hf":
            result = calculate(molecule, basis, method=method)
        else:
            result = calculate(molecule, basis, method="dft", functional=method)
        if method == "hf" and (
            not result.converged or not result.stable
        ):  #! do not check for dft!
            warnings.warn(f"{name} ({backend}, {method}) not converged or stable")
            return
        E_sim = result.electronic_energy() + result.nuclear_repulsion_energy()
        E_ref = refernces[name][method][basis]
        diff = abs(E_sim - E_ref)

        assert diff < tolerance, f"Energy mismatch for {name} ({backend}, {method})"

    except FileNotFoundError:
        warnings.warn(f"Geometry for {name} not found in backend {backend}")


@pytest.mark.parametrize(
    "name, method, basis",
    [
        (name, method, basis)
        for name, entries in refernces.items()
        for method, basis_dict in entries.items()
        for basis in basis_dict
    ],
)
def test_backend_energy_match(context, name: str, method: str, basis: str):
    backends = [Backend.PSI, Backend.PY]

    path = str(Path(__file__).parent / "molecules" / "geometries" / f"{name}.xyz")

    backend_energies = []
    for backend in backends:
        try:
            molecule = load(path, backend)
            if method == "hf":
                result = calculate(molecule, basis, method=method)
            else:
                result = calculate(molecule, basis, method="dft", functional=method)
            backend_energies.append(result.electronic_energy())

            if method == "hf" and (
                not result.converged or not result.stable
            ):  #! do not check for dft!
                warnings.warn(f"{name} ({backend}, {method}) not converged or stable")
                continue

        except FileNotFoundError:
            warnings.warn(f"Geometry for {name} not found in backend {backend}")

    assert similar(*backend_energies, tolerance=1e-3), "backend energies must match"


@pytest.mark.parametrize(
    "name, basis",
    [
        (name, basis)
        for name, entries in refernces.items()
        for _, basis_dict in entries.items()
        for basis in basis_dict
    ],
)
def test_dft_lower_than_hf(context, name: str, basis: str):
    backends = [Backend.PSI, Backend.PY]

    path = str(Path(__file__).parent / "molecules" / "geometries" / f"{name}.xyz")

    for backend in backends:
        try:
            molecule = load(path, backend)
            result_hf = calculate(molecule, basis, method="hf")
            result_dft = calculate(molecule, basis, method="dft", functional="b3lyp")

            if not result_hf.converged or not result_hf.stable:  # check only for hf
                warnings.warn(f"{name} ({backend}, hf) not converged or stable")
                continue

            assert (
                result_dft.electronic_energy() < result_hf.electronic_energy()
            )  # DFT should yield lower energy
            assert similar(
                result_dft.electronic_energy(),
                result_hf.electronic_energy(),
                tolerance=1,
            )  # energies should be similar to a few Hartree
        except FileNotFoundError:
            warnings.warn(f"Geometry for {name} not found in backend {backend}")
