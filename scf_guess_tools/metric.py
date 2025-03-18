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
    S = overlap
    D = tuplify(density)
    F = tuplify(fock)
    E = [f @ d @ S - S @ d @ f for d, f in zip(D, F)]

    return sqrt(sum(e.sum_of_squares for e in E) / sum(e.size for e in E))


@timeable
def energy_error(
    initial_electronic_energy: float, final_electronic_energy: float
) -> float:
    return initial_electronic_energy / final_electronic_energy - 1.0
