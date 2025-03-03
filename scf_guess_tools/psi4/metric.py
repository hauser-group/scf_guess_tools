import numpy as np
import psi4

from .wavefunction import Wavefunction
from psi4.core import Matrix


def f_score(initial: Wavefunction, final: Wavefunction) -> float:
    S = final.S
    D = tuple(zip((*(initial.D,),), (*(final.D,),)))
    Q = [(Di @ S @ Df @ S).trace for Di, Df in D]
    N = [(Df @ S).trace for _, Df in D]

    return sum(Q) / sum(N)


def diis_error(initial: Wavefunction, final: Wavefunction) -> float:
    # TODO this is buggy

    Da_guess, Db_guess = (
        initial.native.Da_subset("AO").np,
        initial.native.Db_subset("AO").np,
    )
    Fa_guess, Fb_guess = (
        initial.native.Fa_subset("AO").np,
        initial.native.Fb_subset("AO").np,
    )
    Da_ref, Db_ref = final.native.Da_subset("AO").np, final.native.Db_subset("AO").np

    S = Matrix(*Da_ref.shape)
    S.remove_symmetry(final.native.S(), final.native.aotoso().transpose())

    Ea = Fa_guess @ Da_guess @ S - S @ Da_guess @ Fa_guess
    Eb = Fb_guess @ Db_guess @ S - S @ Db_guess @ Fb_guess

    return np.trace(Ea @ Ea) + np.trace(Eb @ Eb)


def energy_error(initial: Wavefunction, final: Wavefunction) -> float:
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
                    "REFERENCE": "RHF" if initial.molecule.singlet else "UHF",
                }
            )

            E_guess = psi4.energy("HF", molecule=initial.molecule.native)
            E_ref = final.native.energy()
        finally:
            psi4.core.clean_options()
            psi4.core.clean()
            psi4.core.reopen_outfile()

    return E_guess / E_ref - 1.0
