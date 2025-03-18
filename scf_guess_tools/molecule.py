from __future__ import annotations

from .core import Object
from abc import ABC, abstractmethod
from typing import Any


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
