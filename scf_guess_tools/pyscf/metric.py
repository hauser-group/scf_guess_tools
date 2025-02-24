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
    Da_guess, Db_guess = initial.Da, initial.Db
    Da_ref, Db_ref = final.Da, final.Db
    S = initial.molecule.native.intor("int1e_ovlp")

    Err_a = Da_guess @ Da_ref @ S - S @ Da_guess @ Da_ref
    Err_b = Db_guess @ Db_ref @ S - S @ Db_guess @ Db_ref
    Err_a_t = np.trace(Err_a @ Err_a)
    Err_b_t = np.trace(Err_b @ Err_b)
    return Err_a_t + Err_b_t


def energy_error(initial: Wavefunction, final: Wavefunction) -> float:
    Da_guess, Db_guess = initial.Da, initial.Db
    Da_ref, Db_ref = final.Da, final.Db
    return 2
