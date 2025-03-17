from __future__ import annotations

from .core import Object
from ..molecule import Molecule as Base
from pyscf.gto import M, Mole as Native

import os
import re


class Molecule(Base, Object):
    @property
    def native(self) -> Native:
        return self._native

    @property
    def name(self) -> str:
        return self._name

    @property
    def charge(self) -> int:
        return self._native.charge

    @property
    def multiplicity(self) -> int:
        return self._native.multiplicity

    @property
    def atoms(self) -> int:
        return self._native.natm

    @property
    def geometry(self):
        return self._native.atom

    @property
    def symmetry(self) -> bool:
        return self._native.symmetry

    def __init__(self, name: str, native: Native):
        self._name = name
        self._native = native

    def __getstate__(self):
        return (
            super().__getstate__(),
            self.name,
            self.charge,
            self.multiplicity,
            self.geometry,
            self.symmetry,
        )

    def __setstate__(self, serialized):
        super().__setstate__(serialized[0])
        self._name, q, m, atom, symmetry = serialized[1:]
        self._native = M(atom=atom, charge=q, spin=m - 1, symmetry=symmetry)

    @classmethod
    def load(cls, path: str, symmetry: bool = True) -> Molecule:
        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        with open(path, "r") as file:
            lines = file.readlines()

        q = int(re.search(r"charge\s+(-?\d+)", lines[1]).group(1))
        m = int(re.search(r"multiplicity\s+(\d+)", lines[1]).group(1))

        return Molecule(name, M(atom=path, charge=q, spin=m - 1, symmetry=symmetry))
