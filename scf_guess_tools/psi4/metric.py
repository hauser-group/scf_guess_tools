import numpy as np
import psi4

from .molecule import singlet
from psi4.core import Matrix, Wavefunction


def f_score(guess: Wavefunction, reference: Wavefunction) -> float:
    Da_guess, Db_guess = guess.Da_subset("AO").np, guess.Db_subset("AO").np
    Da_ref, Db_ref = reference.Db_subset("AO").np, reference.Db_subset("AO").np

    S = Matrix(*Da_ref.shape)
    S.remove_symmetry(reference.S(), reference.aotoso().transpose())

    Q = lambda P_guess, P_ref: np.trace(P_guess @ S @ P_ref @ S)
    N = lambda P_ref: np.trace(P_ref @ S)

    numerator = Q(Da_guess, Da_ref) + Q(Db_guess, Db_ref)
    denominator = N(Da_ref) + N(Db_ref)

    return numerator / denominator


def diis_error(guess: Wavefunction, reference: Wavefunction) -> float:
    # TODO this is buggy

    Da_guess, Db_guess = guess.Da_subset("AO").np, guess.Db_subset("AO").np
    Fa_guess, Fb_guess = guess.Fa_subset("AO").np, guess.Fb_subset("AO").np
    Da_ref, Db_ref = reference.Da_subset("AO").np, reference.Db_subset("AO").np

    S = Matrix(*Da_ref.shape)
    S.remove_symmetry(reference.S(), reference.aotoso().transpose())

    Ea = Fa_guess @ Da_guess @ S - S @ Da_guess @ Fa_guess
    Eb = Fb_guess @ Db_guess @ S - S @ Db_guess @ Fb_guess

    return np.trace(Ea @ Ea) + np.trace(Eb @ Eb)


def energy_error(guess: Wavefunction, reference: Wavefunction) -> float:
    # TODO this is buggy
    # TODO this needs generalization to other bases
    # TODO check theory level

    with psi4.driver.p4util.hold_options_state():
        try:
            psi4.core.clean_options()
            psi4.core.clean()
            psi4.core.be_quiet()

            psi4.set_options(
                {
                    "BASIS": "pcseg-0",
                    "SCF_TYPE": "PK",
                    "MAXITER": 0,
                    "FAIL_ON_MAXITER": False,
                    "REFERENCE": "RHF" if singlet(guess.molecule()) else "UHF",
                }
            )

            E_guess = psi4.energy("HF", molecule=guess.molecule())
            E_ref = reference.energy()
        finally:
            psi4.core.clean_options()
            psi4.core.clean()
            psi4.core.reopen_outfile()

    return E_guess / E_ref - 1.0
