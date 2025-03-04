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
    print("DIIIIIIIIIIS")

    print(initial.F.native.to_array(dense=True) - final.F.native.to_array(dense=True))
    print()

    S = final.S.native.to_array(dense=True)
    D = tuple(d.native.to_array(dense=True) for d in tuplify(initial.D))
    F = tuple(f.native.to_array(dense=True) for f in tuplify(final.F))

    E = [f @ d @ S - S @ d @ f for d, f in zip(D, F)]

    # print()
    # print(E)
    # print()

    return np.sqrt(sum([np.sum(e**2) for e in E]) / sum([e.size for e in E]))


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
