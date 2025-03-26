from __future__ import annotations

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


guess_basis = basis_fixture(["sto-3g", "pcseg-0", "pcseg-1"])
guess_path = path_fixture(non_singlet=0, max_atoms=8)

converged_basis = basis_fixture(["pcseg-0"])
converged_path = path_fixture(
    charged=0, non_charged=1, singlet=10, non_singlet=10, max_atoms=15
)

metric_basis = basis_fixture(["pcseg-0"])
metric_path = path_fixture(
    charged=0, non_charged=1, singlet=10, non_singlet=10, max_atoms=15
)


@pytest.mark.parametrize("method", ["hf", "dft"])
def test_core_guess(context, guess_path: str, guess_basis: str, method: str):
    backends = [Backend.PSI, Backend.PY]
    molecules = [load(guess_path, b) for b in backends]
    schemes = ["CORE", "1e"]
    functional = "b3lyp" if method == "dft" else None

    initials = [
        guess(m, guess_basis, s, method=method, functional=functional)
        for m, s in zip(molecules, schemes)
    ]
    ignore = ["molecule", "initial", "time"]
    if method == "dft":
        ignore.extend(["fock", "electronic_energy"])
    assert similar(*initials, ignore=ignore), "core guess wavefunctions must be similar"

    finals = [
        calculate(m, guess_basis, s, method=method, functional=functional)
        for m, s in zip(molecules, schemes)
    ]

    tolerance = 1e-2

    for final in finals:
        if not final.converged or (
            method == "hf" and not final.stable
        ):  # dft doesn't support stability check
            warnings.warn(
                f"Solution for {final.molecule.name} not converged or stable for {method}, skipping"
            )
            return

    f_scores = []
    for initial, final in zip(initials, finals):
        f_scores.append(f_score(initial.overlap(), initial.density(), final.density()))

    assert similar(
        *f_scores, tolerance=tolerance
    ), f"core guesses must have similar f-scores {f_scores}"

    if method == "hf":
        energy_errors = []
        for initial, final in zip(
            initials, finals
        ):  # dft only supports energy after calculation
            energy_errors.append(
                energy_error(initial.electronic_energy(), final.electronic_energy())
            )
        assert similar(
            *energy_errors, tolerance=tolerance
        ), f"core guesses must have similar energy errors {energy_errors}"

        diis_errors = []
        for initial, final in zip(initials, finals):
            diis_errors.append(
                diis_error(initial.overlap(), initial.density(), initial.fock())
            )

        assert similar(
            *diis_errors, tolerance=tolerance
        ), f"core guesses must have similar diis errors {diis_errors}"


@pytest.mark.parametrize("method", ["hf", "dft"])
def test_converged(context, converged_path: str, converged_basis: str, method: str):
    backends = [Backend.PSI, Backend.PY]
    molecules = [load(converged_path, b) for b in backends]
    functional = "b3lyp" if method == "dft" else None
    finals = [
        calculate(m, converged_basis, method=method, functional=functional)
        for m in molecules
    ]
    to_ignore = ["molecule", "initial", "time", "stable", "second_order"]

    for final in finals:
        if not final.converged or (
            method == "hf" and not final.stable
        ):  # dft doesn't support stability check
            warnings.warn(
                f"Solution for {final.molecule.name} not converged or stable for {method}, skipping"
            )
            return

        f = f_score(final.overlap(), final.density(), final.density())
        assert 1 - f < 1e-10, "converged f-score must be close to 1"

        e = energy_error(final.electronic_energy(), final.electronic_energy())
        assert e < 1e-10, "energy error must be close to 0"

        if method == "hf":  # fock is not available for dft
            d = diis_error(final.overlap(), final.density(), final.fock())
            assert d < 1e-5, "diis error must be close to 0"
    if method == "dft":
        to_ignore.extend(["fock"])
    assert similar(*finals, ignore=to_ignore), "converged wavefunctions must be similar"


@pytest.mark.parametrize(
    "backend, metric, method",
    [
        (backend, metric, method)
        for backend in [Backend.PSI, Backend.PY]
        for metric in [f_score, diis_error, energy_error]
        for method in ["hf", "dft"]
    ],
)
def test_metric(
    context,
    backend: Backend,
    metric_path: str,
    metric_basis: str,
    metric: Callable,
    method: str,
):
    functional = "b3lyp" if method == "dft" else None

    molecule = load(metric_path, backend)
    final = calculate(molecule, metric_basis, method=method, functional=functional)

    if not final.converged or (
        method == "hf" and not final.stable
    ):  # dft doesn't support stability check
        warnings.warn(
            f"Solution for {final.molecule.name} not converged or stable for {method}, skipping"
        )
        return

    scores = set()
    for scheme in guessing_schemes(backend):
        initial = guess(
            molecule, metric_basis, scheme, method=method, functional=functional
        )

        if metric == f_score:
            scores.add(f_score(initial.overlap(), initial.density(), final.density()))
        elif metric == diis_error and method == "hf":
            scores.add(diis_error(initial.overlap(), initial.density(), initial.fock()))
        elif (
            metric == energy_error and method == "hf"
        ):  # dft doesn't support energy before calculation - in initial
            scores.add(
                energy_error(initial.electronic_energy(), final.electronic_energy())
            )
        else:
            if method == "dft":
                warnings.warn(f"Skipping metric {metric} for dft")
                continue
            assert f"Unexpected metric {metric}"
    if method == "hf":
        assert len(scores) > 1, "different schemes must yield different scores"


@pytest.mark.parametrize(
    "backend, method",
    [
        (backend, method)
        for backend in [Backend.PSI, Backend.PY]
        for method in ["hf", "dft"]
    ],
)
def test_f_score(
    context, backend: Backend, metric_path: str, metric_basis: str, method: str
):
    functional = "b3lyp" if method == "dft" else None
    molecule = load(metric_path, backend)
    final = calculate(molecule, metric_basis, method=method, functional=functional)

    S = final.overlap()
    Df = final.density()

    DfS = tuple(df @ S for df in tuplify(Df))
    DfS = DfS if isinstance(Df, tuple) else DfS[0]

    if not final.converged or (
        method == "hf" and not final.stable
    ):  # dft doesn't support stability check
        warnings.warn(
            f"Solution for {final.molecule.name} not converged or stable for {method}, skipping"
        )
        return

    for scheme in guessing_schemes(backend):
        initial = guess(
            molecule, metric_basis, scheme, method=method, functional=functional
        )

        regular = f_score(S, initial.density(), Df)
        boosted = f_score(S, initial.density(), DfS, skip_final_overlap=True)

        assert np.isclose(
            regular, boosted, rtol=1e-10
        ), f"regular and boosted f-score must be equal {regular} {boosted}"


@pytest.mark.parametrize(
    "backend",
    [(backend) for backend in [Backend.PSI, Backend.PY]],
)
def test_hf_vs_dft(
    context, converged_path: str, converged_basis: str, backend: Backend
):
    molecule = load(converged_path, backend)
    hf_final = calculate(molecule, converged_basis, method="hf")
    dft_final = calculate(molecule, converged_basis, method="dft", functional="b3lyp")

    if any(
        not final.converged or (method == "hf" and not final.stable)
        for final, method in zip([hf_final, dft_final], ["hf", "dft"])
    ):  # dft doesn't support stability check
        warnings.warn(
            f"Solution for {hf_final.molecule.name} not converged or stable, skipping"
        )
        return

    # tolerance has to be higher here than default because dft and hf densities differ a bit (because dft yields solutions with lower energy)
    assert similar(
        hf_final,
        dft_final,
        tolerance=1e-4,
        ignore=["fock", "time", "stable", "second_order"],
    ), "hf and dft wavefunctions must be similar"
