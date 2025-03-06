from abc import ABC, abstractmethod
from joblib import Memory
import os


class Engine(ABC):
    def __init__(self, cache: str, verbose: int):
        if cache is None:
            return

        base = os.environ.get("SGT_CACHE")

        if base is None:
            raise RuntimeError("SGT_CACHE environment variable not set")

        self._memory = Memory(base, verbose=verbose)

        self.guess = self._memory.cache(self.guess, ignore=["self"])
        self.calculate = self._memory.cache(self.calculate, ignore=["self"])
        self.score = self._memory.cache(self.score, ignore=["self"])

    @classmethod
    @abstractmethod
    def __repr__(cls) -> str:
        pass

    @classmethod
    @abstractmethod
    def guessing_schemes(cls) -> list[str]:
        pass

    @property
    def memory(self) -> Memory:
        return self._memory
