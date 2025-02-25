import numpy as np
import pyscf.scf as scf
from .wavefunction import Wavefunction


def f_score(initial: Wavefunction, final: Wavefunction) -> float:
    Da_guess, Db_guess = initial.Da, initial.Db
    Da_ref, Db_ref = final.Da, final.Db

    S = initial.molecule.native.intor("int1e_ovlp")

    Q = lambda P_guess, P_ref: np.sum(P_guess * (S @ P_ref @ S))
    N = lambda P_ref: np.sum(P_ref * S)

    numerator = Q(Da_guess, Da_ref) + Q(Db_guess, Db_ref)
    denominator = N(Da_ref) + N(Db_ref)

    return numerator / denominator


def diis_error(initial: Wavefunction, final: Wavefunction) -> float:
    # TODO look into the fro.-norm -> if there is some normalization to be done thereafter for triplets
    Da_guess, Db_guess = initial.Da, initial.Db

    S = initial.molecule.native.intor("int1e_ovlp")

    if initial.molecule.singlet:
        Fa_guess = scf.hf.get_fock(scf.RHF(initial.molecule.native), dm=Da_guess)
        Fb_guess = Fa_guess
    else:
        Fa_guess, Fb_guess = scf.hf.get_fock(
            scf.UHF(initial.molecule.native), dm=(Da_guess, Db_guess)
        )

    Err_a = Da_guess @ Fa_guess @ S - S @ Da_guess @ Fa_guess
    Err_b = Db_guess @ Fb_guess @ S - S @ Db_guess @ Fb_guess

    if Err_a.ndim == 2:  # singlet!
        Err_a_t = np.trace(Err_a @ Err_a)
        Err_b_t = np.trace(Err_b @ Err_b)

    # UHF
    elif Err_a.ndim == 3 and Err_a.shape[0] == 2:
        Err_a_t = np.linalg.norm(Err_a[0], "fro") + np.linalg.norm(Err_a[1], "fro")
        Err_b_t = np.linalg.norm(Err_b[0], "fro") + np.linalg.norm(Err_b[1], "fro")

    else:
        raise ValueError(f"Unexpected shape for error matrices: {Err_a.shape}")

    return Err_a_t + Err_b_t


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
