from __future__ import annotations

from .common import timeable
from .core import Backend, Object
from .proxy import proxy
from abc import ABC, abstractmethod
from functools import wraps
from typing import Any

import joblib


class Molecule(Object, ABC):
    """Provides a common interface for molecules across different backends."""

    @property
    @abstractmethod
    def native(self) -> Any:
        """The underlying backend-specific molecular object."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the molecule."""
        pass

    @property
    @abstractmethod
    def charge(self) -> int:
        """Total charge of the molecule."""
        pass

    @property
    @abstractmethod
    def multiplicity(self) -> int:
        """Spin multiplicity of the molecule."""
        pass

    @property
    def singlet(self) -> bool:
        """Whether the molecule is a singlet."""
        return self.multiplicity == 1

    @property
    @abstractmethod
    def atoms(self) -> int:
        """Number of atoms in the molecule."""
        pass

    @property
    @abstractmethod
    def geometry(self) -> Any:
        """Molecular geometry in the backend's native format."""
        pass

    @property
    @abstractmethod
    def symmetry(self) -> bool:
        """Whether symmetry is enabled."""
        pass

    def __hash__(self) -> int:
        """Return a deterministic hash.

        Returns:
            A hash value uniquely identifying the molecule.
        """
        return int(joblib.hash((self.backend(), self.__getstate__())), 16)

    @classmethod
    @abstractmethod
    @timeable
    def load(cls, path: str, symmetry: bool = True, **kwargs) -> Molecule:
        """Load a molecule from an xyz file.

        Args:
            path: The path to the xyz file.
            symmetry: Whether to enable symmetry.
            **kwargs: Additional backend-specific keyword arguments.

        Returns:
            The loaded molecule.
        """
        pass


@wraps(Molecule.load)
def load(path: str, backend: Backend, **kwargs):
    return proxy(backend, lambda p: p.Molecule.load, path, **kwargs)
