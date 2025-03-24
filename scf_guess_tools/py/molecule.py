from __future__ import annotations

from ..common import timeable
from ..molecule import Molecule as Base
from .core import Object
from pyscf.gto import M, Mole as Native

import os
import re


class Molecule(Base, Object):
    """Molecular representation using the PySCF backend. This class provides an
    implementation of the Molecule interface using PySCF's native molecular object.
    """

    @property
    def native(self) -> Native:
        """The underlying PySCF molecular object."""
        return self._native

    @property
    def name(self) -> str:
        """The name of the molecule."""
        return self._name

    @property
    def charge(self) -> int:
        """Total charge of the molecule."""
        return self._native.charge

    @property
    def multiplicity(self) -> int:
        """Spin multiplicity of the molecule."""
        return self._native.multiplicity

    @property
    def atoms(self) -> int:
        """Number of atoms in the molecule."""
        return self._native.natm

    @property
    def geometry(self):
        """Molecular geometry in PySCF's native format."""
        return self._native.atom

    @property
    def symmetry(self) -> bool:
        """Whether symmetry is enabled."""
        return self._native.symmetry

    def __init__(self, name: str, native: Native):
        """Initialize the PySCF molecular object.

        Args:
            name: The name of the molecule.
            native: The PySCF Mole instance.
        """
        self._name = name
        self._native = native

    def __getstate__(self):
        """Serialize the molecule for pickling."""
        return (
            super().__getstate__(),
            self.name,
            self.charge,
            self.multiplicity,
            self.geometry,
            self.symmetry,
        )

    def __setstate__(self, serialized):
        """Restore the molecule from a serialized state.

        Args:
            serialized: The serialized molecule data.
        """
        super().__setstate__(serialized[0])
        self._name, q, m, atom, symmetry = serialized[1:]
        self._native = M(atom=atom, charge=q, spin=m - 1, symmetry=symmetry)

    @classmethod
    @timeable
    def load(cls, path: str, symmetry: bool = True) -> Molecule:
        """Load a molecule from an xyz file.

        Args:
            path: The path to the xyz file.
            symmetry: Whether to enable symmetry.

        Returns:
            The loaded molecule.
        """
        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        with open(path, "r") as file:
            lines = file.readlines()

        q = int(re.search(r"charge\s+(-?\d+)", lines[1]).group(1))
        m = int(re.search(r"multiplicity\s+(\d+)", lines[1]).group(1))

        native = M(atom=path, charge=q, spin=m - 1, symmetry=symmetry)
        native.atom = native._atom

        return Molecule(name, native)
