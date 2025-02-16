from .wavefunction import Wavefunction


def f_score(initial: Wavefunction, final: Wavefunction) -> float:
    return 0


def diis_error(initial: Wavefunction, final: Wavefunction) -> float:
    return 1


def energy_error(initial: Wavefunction, final: Wavefunction) -> float:
    return 2
