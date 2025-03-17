from __future__ import annotations

from common import equal, replace_random_digit, similar
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


def test_core_guess(context, guess_path: str, guess_basis: str):
    backends = [Backend.PSI, Backend.PY]
    molecules = [load(guess_path, b) for b in backends]
    schemes = ["CORE", "1e"]
    initials = [guess(m, guess_basis, s) for m, s in zip(molecules, schemes)]

    assert similar(
        *initials, ignore=["molecule", "initial", "time"]
    ), "core guess wavefunctions must be similar"

    finals = [calculate(m, guess_basis, s) for m, s in zip(molecules, schemes)]
    tolerance = 1e-2

    for final in finals:
        if not final.converged or not final.stable:
            warnings.warn(
                f"Solution for {final.molecule.name} not converged or stable, skipping"
            )
            return

    f_scores = [f_score(i, f) for i, f in zip(initials, finals)]
    assert similar(
        *f_scores, tolerance=tolerance
    ), f"core guesses must have similar f-scores {f_scores}"

    diis_errors = [diis_error(i) for i in initials]
    assert similar(
        *diis_errors, tolerance=tolerance
    ), f"core guesses must have similar diis errors {diis_errors}"

    energy_errors = [energy_error(i, f) for i, f in zip(initials, finals)]
    assert similar(
        *energy_errors, tolerance=tolerance
    ), f"core guesses must have similar energy errors {energy_errors}"


def test_converged(context, converged_path: str, converged_basis: str):
    backends = [Backend.PSI, Backend.PY]
    molecules = [load(converged_path, b) for b in backends]
    finals = [calculate(m, converged_basis) for m in molecules]

    for final in finals:
        if not final.converged or not final.stable:
            warnings.warn(
                f"Solution for {final.molecule.name} not converged or stable, skipping"
            )
            return

        f = f_score(final, final)
        assert 1 - f < 1e-10, "converged f-score must be close to 1"

        d = diis_error(final)
        assert d < 1e-5, "diis error must be close to 0"

        e = energy_error(final, final)
        assert e < 1e-10, "energy error must be close to 0"

    assert similar(
        *finals, ignore=["molecule", "initial", "time", "stable", "second_order"]
    ), "converged wavefunctions must be similar"


@pytest.mark.parametrize(
    "backend, metric",
    [
        (backend, metric)
        for backend in [Backend.PSI, Backend.PY]
        for metric in [f_score, diis_error, energy_error]
    ],
)
def test_metric(
    context, backend: Backend, metric_path: str, metric_basis: str, metric: Callable
):
    molecule = load(metric_path, backend)
    final = calculate(molecule, metric_basis)

    if not final.converged or not final.stable:
        warnings.warn(f"Solution for {molecule.name} not converged or stable, skipping")
        return

    scores = set()
    for scheme in guessing_schemes(backend):
        initial = guess(molecule, metric_basis, scheme)

        if metric == f_score:
            scores.add(f_score(initial, final))
        elif metric == diis_error:
            scores.add(diis_error(initial))
        elif metric == energy_error:
            scores.add(energy_error(initial, final))
        else:
            assert f"Unexpected metric {metric}"

    assert len(scores) > 1, "different schemes must yield different scores"


@pytest.mark.parametrize("backend", [Backend.PSI, Backend.PY])
def test_f_score(context, backend: Backend, metric_path: str, metric_basis: str):
    molecule = load(metric_path, backend)
    final = calculate(molecule, metric_basis)

    S = final.overlap()
    DfS = tuple(Df @ S for Df in final.density(tuplify=True))

    if not final.converged or not final.stable:
        warnings.warn(f"Solution for {molecule.name} not converged or stable, skipping")
        return

    for scheme in guessing_schemes(backend):
        initial = guess(molecule, metric_basis, scheme)

        regular = f_score(initial, final)
        boosted = f_score(initial, DfS=DfS)

        assert np.isclose(
            regular, boosted, rtol=1e-10
        ), f"regular and boosted f-score must be equal {regular} {boosted}"
