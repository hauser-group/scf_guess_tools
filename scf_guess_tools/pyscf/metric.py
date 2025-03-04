from ..common import tuplify
from .wavefunction import Wavefunction

import numpy as np
import pyscf.scf as scf


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
    if initial.molecule.singlet:
        mf = scf.RHF(initial.molecule.native)
        initial_D = 2 * initial.Da
        final_D = 2 * final.Da
    else:
        mf = scf.UHF(initial.molecule.native)
        initial_D = (initial.Da, initial.Db)
        final_D = (final.Da, final.Db)

    hcore = mf.get_hcore(initial.molecule.native)

    # energy initial
    initial_D = 2 * initial.Da
    veff_guess = mf.get_veff(initial.molecule.native, initial_D)
    E_guess_elec, _ = mf.energy_elec(initial_D, hcore, veff_guess)
    E_guess = E_guess_elec + initial.molecule.native.energy_nuc()

    # energy final
    final_D = 2 * final.Da
    veff_ref = mf.get_veff(final.molecule.native, final_D)
    E_ref_elec, _ = mf.energy_elec(final_D, hcore, veff_ref)
    E_ref = E_ref_elec + final.molecule.native.energy_nuc()

    return E_guess / E_ref - 1.0
