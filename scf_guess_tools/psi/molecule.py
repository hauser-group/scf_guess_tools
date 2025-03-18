from __future__ import annotations

from ..molecule import Molecule as Base
from .core import Object
from psi4.core import Molecule as Native
from typing import overload

import os
import re


class Molecule(Base, Object):
    """Molecular representation using the Psi4 backend. This class provides an
    implementation of the Molecule interface using Psi4's native molecular object.
    """

    @property
    def native(self) -> Native:
        """The underlying Psi4 molecular object."""
        return self._native

    @property
    def name(self) -> str:
        """The name of the molecule."""
        return self._native.name()

    @property
    def charge(self) -> int:
        """Total charge of the molecule."""
        return self._native.molecular_charge()

    @property
    def multiplicity(self) -> int:
        """Spin multiplicity of the molecule."""
        return self._native.multiplicity()

    @property
    def atoms(self) -> int:
        """Number of atoms in the molecule."""
        return self._native.natom()

    @property
    def geometry(self):
        """Molecular geometry in Psi4's native format."""
        return self._native.to_string(dtype="psi4")

    @property
    def symmetry(self) -> bool:
        """Whether symmetry is enabled."""
        return self._symmetry

    def __init__(self, native: Native, symmetry: bool):
        """Initialize the Psi4 molecular object.

        Args:
            native: The Psi4 Molecule instance.
            symmetry: Whether symmetry is enabled.
        """
        self._native = native
        self._symmetry = symmetry

    def __getstate__(self):
        """Serialize the molecule for pickling."""
        return super().__getstate__(), self.name, self.geometry, self.symmetry

    def __setstate__(self, serialized):
        """Restore the molecule from a serialized state.

        Args:
            serialized: The serialized molecule data.
        """
        super().__setstate__(serialized[0])

        self._native = Native.from_string(
            serialized[2], name=serialized[1], dtype="psi4"
        )

        self._symmetry = serialized[3]

    @classmethod
    def load(cls, path: str, symmetry: bool = True) -> Molecule:
        """Load a molecule from an xyz file.

        Args:
            path: The path to the xyz file.
            symmetry: Whether to enable symmetry.

        Returns:
            The loaded molecule.
        """
        with open(path, "r") as file:
            lines = file.readlines()

        q = re.search(r"charge\s+(-?\d+)", lines[1]).group(1)
        m = re.search(r"multiplicity\s+(\d+)", lines[1]).group(1)

        lines[1] = f"{q} {m}\n"
        xyz = "".join(lines)

        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        molecule = Native.from_string(xyz, name=name, dtype="xyz+")

        if not symmetry:
            molecule.reset_point_group("C1")

        return Molecule(molecule, symmetry)
