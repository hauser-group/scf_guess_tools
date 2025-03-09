from __future__ import annotations

from .common import tuplify
from .wavefunction import Wavefunction
from abc import ABC, abstractmethod
from math import sqrt
from time import process_time


class Metric(ABC):
    def __init__(
        self,
        initial: Wavefunction,
        final: Wavefunction | None = None,
        enforce_final=True,
    ):
        self._initial = initial
        self._final = final

        if enforce_final and final is None:
            calculate = initial.__class__.engine().calculate
            self._final = calculate(initial.molecule, initial.basis)

        start = process_time()
        self._value = self.__call__()
        self._time = process_time() - start

    @property
    def initial(self) -> Wavefunction:
        return self._initial

    @property
    def final(self) -> Wavefunction | None:
        return self._final

    @property
    def time(self) -> float | None:
        return self._time

    @property
    def value(self) -> float | None:
        return self._value

    @abstractmethod
    def __call__(self) -> float:
        pass

    @classmethod
    @abstractmethod
    def __repr__(cls) -> str:
        pass


class FScore(Metric):
    def __init__(self, initial: Wavefunction, final: Wavefunction | None = None):
        super().__init__(initial, final, enforce_final=True)

    def __call__(self) -> float:
        S = self.final.S
        D = tuple(zip(tuplify(self.initial.D), tuplify(self.final.D)))
        Q = [(Di @ S @ Df @ S).trace for Di, Df in D]
        N = [(Df @ S).trace for _, Df in D]

        return sum(Q) / sum(N)

    @classmethod
    def __repr__(cls) -> str:
        return "F-Score"


class DIISError(Metric):
    def __init__(self, initial: Wavefunction, *args, **kwargs):
        super().__init__(initial, enforce_final=False)

    def __call__(self) -> float:
        S = self.initial.S
        D = tuplify(self.initial.D)
        F = tuplify(self.initial.F)
        E = [f @ d @ S - S @ d @ f for d, f in zip(D, F)]

        return sqrt(sum(e.sum_of_squares for e in E) / sum(e.size for e in E))

    @classmethod
    def __repr__(cls) -> str:
        return "DIIS Error"


class EnergyError(Metric):
    def __init__(self, initial: Wavefunction, final: Wavefunction | None = None):
        super().__init__(initial, final, enforce_final=True)

    def __call__(self) -> float:
        return self.initial.energy / self.final.energy - 1.0

    @classmethod
    def __repr__(cls) -> str:
        return "Energy Error"
