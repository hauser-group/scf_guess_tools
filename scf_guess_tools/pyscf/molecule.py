from __future__ import annotations

from ..molecule import Molecule as Base
from pyscf.gto import M, Mole as Native

import os
import re
import scf_guess_tools.pyscf.engine as e


class Molecule(Base):
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

    def __init__(self, engine: e.Engine, name: str, native: Native):
        super().__init__(engine)
        self._name = name
        self._native = native

    def __getstate__(self):
        return self.name, self.charge, self.multiplicity, self.geometry

    def __setstate__(self, serialized):
        self._name, q, m, atom = serialized
        self._native = M(atom=atom, charge=q, spin=m - 1)

    @classmethod
    def load(cls, engine: e.Engine, path: str) -> Molecule:
        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        with open(path, "r") as file:
            lines = file.readlines()

        q = int(re.search(r"charge\s+(-?\d+)", lines[1]).group(1))
        m = int(re.search(r"multiplicity\s+(\d+)", lines[1]).group(1))

        return Molecule(engine, name, M(atom=path, charge=q, spin=m - 1))
