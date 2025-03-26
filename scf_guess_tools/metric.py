from __future__ import annotations

from .common import timeable, tuplify
from .matrix import Matrix
from math import sqrt


@timeable
def f_score(
    overlap: Matrix,
    initial_density: Matrix | tuple[Matrix, Matrix],
    final_density: Matrix | tuple[Matrix, Matrix],
    skip_final_overlap: bool = False,
) -> float:
    """Compute the fraction of electron density covered by the initial density as
     defined by Lehtola (2019, DOI: https://pubs.acs.org/doi/10.1021/acs.jctc.8b01089).

    Args:
        overlap: The overlap matrix.
        initial_density: The initial density matrix (RHF) or a tuple of alpha and beta
            matrices (UHF).
        final_density: The final density matrix (RHF) or a tuple of alpha and beta
            matrices (UHF).
        skip_final_overlap: If True, assume final_density is pre-multiplied by overlap.

    Returns:
        The computed f-score.
    """
    S = overlap
    DiS = [di @ S for di in tuplify(initial_density)]
    DfS = [df if skip_final_overlap else df @ S for df in tuplify(final_density)]

    Q = [(dis @ dfs).trace for dis, dfs in zip(DiS, DfS)]
    N = [dfs.trace for dfs in DfS]

    return sum(Q) / sum(N)


@timeable
def diis_error(
    overlap: Matrix,
    density: Matrix | tuple[Matrix, Matrix],
    fock: Matrix | tuple[Matrix, Matrix],
) -> float:
    """Compute the RMS of the DIIS error vector.

    Args:
        overlap: The overlap matrix.
        density: The density matrix (RHF) or a tuple of alpha and beta matrices (UHF).
        fock: The Fock matrix (RHF) or a tuple of alpha and beta matrices (UHF).

    Returns:
        The RMS of the DIIS error vector.
    """
    S = overlap
    D = tuplify(density)
    F = tuplify(fock)
    E = [f @ d @ S - S @ d @ f for d, f in zip(D, F)]

    return sqrt(sum(e.sum_of_squares for e in E) / sum(e.size for e in E))


@timeable
def energy_error(
    initial_electronic_energy: float, final_electronic_energy: float
) -> float:
    """Compute the relative error between initial and converged electronic energy.

    Args:
        initial_electronic_energy: An arbitrary electronic energy estimate.
        final_electronic_energy: The converged electronic energy.

    Returns:
        The relative error.
    """
    return initial_electronic_energy / final_electronic_energy - 1.0
