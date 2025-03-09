from __future__ import annotations

from common import similar
from provider import context, engine, basis_fixture, path_fixture
from scf_guess_tools import (
    Engine,
    Matrix,
    Metric,
    PyEngine,
    PsiEngine,
    FScore,
    DIISError,
    EnergyError,
)

import numpy as np
import pytest

py_schemes = ["minao", "1e", "atom", "huckel", "vsap"]
psi_schemes = ["CORE", "SAD", "GWH", "HUCKEL", "SAP"]

guess_basis = basis_fixture(["sto-3g", "pcseg-0", "pcseg-2"])
guess_path = path_fixture(paths=10)

calculate_basis = basis_fixture(["pcseg-1"])
calculate_path = path_fixture(paths=5, large=False)

metric_basis = basis_fixture(["sto-3g", "pcseg-1"])
metric_path = path_fixture(paths=5, large=False)


def test_guess(context, guess_path: str, guess_basis: str):
    engines = [PyEngine(cache=False, verbose=1), PsiEngine(cache=False, verbose=1)]
    molecules = [e.load(guess_path) for e in engines]
    schemes = ["1e", "CORE"]
    guesses = [
        e.guess(m, guess_basis, s) for e, m, s in zip(engines, molecules, schemes)
    ]

    assert similar(
        *guesses, ignore=["molecule", "time", "initial"]
    ), "core guesses have to be similar"


@pytest.mark.parametrize("schemes", list(zip(py_schemes, psi_schemes)))
def test_calculate(
    context, calculate_path: str, calculate_basis, schemes: tuple[str, str]
):
    engines = [PyEngine(cache=False, verbose=1), PsiEngine(cache=False, verbose=1)]
    molecules = [e.load(calculate_path) for e in engines]
    wavefunctions = [
        e.calculate(m, calculate_basis, s)
        for e, m, s in zip(engines, molecules, schemes)
    ]

    assert similar(
        *wavefunctions,
        ignore=["molecule", "time", "initial", "iterations", "retried"],
    ), "converged wavefunctions have to be similar"


@pytest.mark.parametrize("metric", [FScore, DIISError, EnergyError])
def test_metric(context, metric_path: str, metric_basis: str, metric: Metric):
    engines = [PyEngine(cache=False, verbose=1), PsiEngine(cache=False, verbose=1)]
    molecules = [e.load(metric_path) for e in engines]
    schemes = ["1e", "CORE"]
    guesses = [
        e.guess(m, metric_basis, s) for e, m, s in zip(engines, molecules, schemes)
    ]
    metrics = [metric(g) for g in guesses]

    assert similar(*[m.value for m in metrics]), "metric values have to be similar"
