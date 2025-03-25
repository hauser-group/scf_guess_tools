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
    initials = [
        guess(m, guess_basis, s, method=method) for m, s in zip(molecules, schemes)
    ]
    ignore = ["molecule", "initial", "time"]
    if method == "dft":
        ignore.extend(["fock", "electronic_energy"])
    assert similar(*initials, ignore=ignore), "core guess wavefunctions must be similar"

    finals = [
        calculate(m, guess_basis, s, method=method) for m, s in zip(molecules, schemes)
    ]
    tolerance = 1e-2

    for final in finals:
        if not final.converged or not final.stable:
            warnings.warn(
                f"Solution for {final.molecule.name} not converged or stable, skipping"
            )
            return

    f_scores = []
    for initial, final in zip(initials, finals):
        f_scores.append(f_score(initial.overlap(), initial.density(), final.density()))

    assert similar(
        *f_scores, tolerance=tolerance
    ), f"core guesses must have similar f-scores {f_scores}"

    energy_errors = []
    for initial, final in zip(initials, finals):
        energy_errors.append(
            energy_error(initial.electronic_energy(), final.electronic_energy())
        )
    assert similar(
        *energy_errors, tolerance=tolerance
    ), f"core guesses must have similar energy errors {energy_errors}"

    if method == "hf":
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
    finals = [calculate(m, converged_basis, method=method) for m in molecules]

    for final in finals:
        if not final.converged or not final.stable:
            warnings.warn(
                f"Solution for {final.molecule.name} not converged or stable, skipping"
            )
            return

        f = f_score(final.overlap(), final.density(), final.density())
        assert 1 - f < 1e-10, "converged f-score must be close to 1"

        e = energy_error(final.electronic_energy(), final.electronic_energy())
        assert e < 1e-10, "energy error must be close to 0"

        if method == "hf":
            d = diis_error(final.overlap(), final.density(), final.fock())
            assert d < 1e-5, "diis error must be close to 0"

    assert similar(
        *finals, ignore=["molecule", "initial", "time", "stable", "second_order"]
    ), "converged wavefunctions must be similar"


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
    molecule = load(metric_path, backend)
    final = calculate(molecule, metric_basis, method=method)

    if (
        not final.converged or not final.stable
    ):  # dft not checked here bc everything would be skipped
        warnings.warn(f"Solution for {molecule.name} not converged or stable, skipping")
        return

    scores = set()
    for scheme in guessing_schemes(backend):
        initial = guess(molecule, metric_basis, scheme, method=method)

        if metric == f_score:
            scores.add(f_score(initial.overlap(), initial.density(), final.density()))
        elif metric == diis_error and method == "hf":
            scores.add(diis_error(initial.overlap(), initial.density(), initial.fock()))
        elif metric == energy_error:
            scores.add(
                energy_error(initial.electronic_energy(), final.electronic_energy())
            )
        else:
            if method == "dft":
                warnings.warn(f"Skipping metric {metric} for dft")
                continue
            assert f"Unexpected metric {metric}"

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
    molecule = load(metric_path, backend)
    final = calculate(molecule, metric_basis, method=method)

    S = final.overlap()
    Df = final.density()

    DfS = tuple(df @ S for df in tuplify(Df))
    DfS = DfS if isinstance(Df, tuple) else DfS[0]

    if not final.converged or not final.stable:
        warnings.warn(f"Solution for {molecule.name} not converged or stable, skipping")
        return

    for scheme in guessing_schemes(backend):
        initial = guess(molecule, metric_basis, scheme, method=method)

        regular = f_score(S, initial.density(), Df)
        boosted = f_score(S, initial.density(), DfS, skip_final_overlap=True)

        assert np.isclose(
            regular, boosted, rtol=1e-10
        ), f"regular and boosted f-score must be equal {regular} {boosted}"
