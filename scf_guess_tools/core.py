from abc import ABC, abstractmethod
from enum import Enum

import os


class Backend(Enum):
    PSI = "Psi"
    PY = "Py"


class Object(ABC):
    def __getstate__(self):
        return self.backend()

    def __setstate__(self, serialized):
        assert serialized == self.backend()

    @classmethod
    @abstractmethod
    def backend(self) -> Backend:
        pass


cache_directory = os.environ.get("SGT_CACHE")
