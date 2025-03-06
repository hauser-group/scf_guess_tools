import numpy as np
import psi4

from ..common import tuplify
from .wavefunction import Wavefunction
from psi4.core import Matrix


def f_score(initial: Wavefunction, final: Wavefunction) -> float:
    S = final.S
    D = tuple(zip(tuplify(initial.D), tuplify(final.D)))
    Q = [(Di @ S @ Df @ S).trace for Di, Df in D]
    N = [(Df @ S).trace for _, Df in D]

    return sum(Q) / sum(N)


def diis_error(initial: Wavefunction, final: Wavefunction) -> float:
    S = initial.S
    D = tuplify(initial.D)
    F = tuplify(initial.F)
    E = [f @ d @ S - S @ d @ f for d, f in zip(D, F)]

    return np.sqrt(sum([e.sum_of_squares for e in E]) / sum([e.size for e in E]))


def energy_error(initial: Wavefunction, final: Wavefunction) -> float:
    return initial.energy / final.energy - 1.0
