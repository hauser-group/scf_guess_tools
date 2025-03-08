from common import similar
from provider import context, engine, path, basis
from scf_guess_tools.common import tuplify
from scf_guess_tools import (
    Engine,
    Matrix,
    PyEngine,
    PsiEngine,
    FScore,
    DIISError,
    EnergyError,
)

import numpy as np
import pytest


def test_guess(context, path: str, basis: str):
    engines = [PyEngine(cache=False, verbose=1), PsiEngine(cache=False, verbose=1)]
    molecules = [e.load(path) for e in engines]
    schemes = ["1e", "CORE"]
    guesses = [e.guess(m, basis, s) for e, m, s in zip(engines, molecules, schemes)]

    assert similar(
        *guesses, ignore=["molecule", "time", "initial"]
    ), "core guesses have to be similar"


@pytest.mark.parametrize(
    "schemes",
    list(
        zip(
            ["minao", "1e", "atom", "huckel", "vsap"],
            ["CORE", "SAD", "GWH", "HUCKEL", "SAP"],
        )
    ),
)
def test_calculate(context, path: str, schemes: tuple[str, str]):
    engines = [PyEngine(cache=False, verbose=1), PsiEngine(cache=False, verbose=1)]
    molecules = [e.load(path) for e in engines]
    wavefunctions = [
        e.calculate(m, "pcseg-1", s) for e, m, s in zip(engines, molecules, schemes)
    ]

    assert similar(
        *wavefunctions,
        ignore=["molecule", "time", "initial", "iterations", "retried"],
    ), "converged wavefunctions have to be similar"
