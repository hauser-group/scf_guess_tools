import numpy as np

from .wavefunction import Wavefunction


def f_score(initial: Wavefunction, final: Wavefunction) -> float:
    Da_guess, Db_guess = initial.Da, initial.Db
    Da_ref, Db_ref = final.Da, final.Db

    S = initial.molecule.native.intor("int1e_ovlp")

    Q = lambda P_guess, P_ref: np.trace(P_guess @ S @ P_ref @ S)
    N = lambda P_ref: np.trace(P_ref @ S)

    numerator = Q(Da_guess, Da_ref) + Q(Db_guess, Db_ref)
    denominator = N(Da_ref) + N(Db_ref)

    return numerator / denominator


def diis_error(initial: Wavefunction, final: Wavefunction) -> float:
    return 1


def energy_error(initial: Wavefunction, final: Wavefunction) -> float:
    return 2
