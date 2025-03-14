from __future__ import annotations

from .common import timeable, tuplify
from .wavefunction import Wavefunction
from .matrix import Matrix
from math import sqrt


@timeable
def f_score(
    initial: Wavefunction,
    final: Wavefunction | None = None,
    final_density_mul_overlap: Matrix | None = None,
) -> float:
    assert final is not None or final_density_mul_overlap is not None

    S = initial.overlap()

    if final_density_mul_overlap is None:
        DfS = [df @ S for df in final.density(tuplify=True)]
    else:
        DfS = tuplify(final_density_mul_overlap)

    Q = [(di @ S @ dfs).trace for di, dfs in zip(initial.density(tuplify=True), DfS)]
    N = [dfs.trace for dfs in DfS]

    return sum(Q) / sum(N)


@timeable
def diis_error(
    initial: Wavefunction,
    final: Wavefunction | None = None,
    use_final_fock: bool = False,
) -> float:

    S = initial.overlap()
    D = initial.density(tuplify=True)
    F = final.fock(tuplify=True) if use_final_fock else initial.fock(tuplify=True)
    E = [f @ d @ S - S @ d @ f for d, f in zip(D, F)]

    return sqrt(sum(e.sum_of_squares for e in E) / sum(e.size for e in E))


@timeable
def energy_error(
    initial: Wavefunction, final: Wavefunction, use_final_fock: bool = False
) -> float:
    F = final.fock() if use_final_fock else initial.fock()
    return initial.electronic_energy(fock=F) / final.electronic_energy(fock=F) - 1.0
