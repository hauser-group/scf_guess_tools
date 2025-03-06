from __future__ import annotations

from ..molecule import Molecule as Base
from psi4.core import Molecule as Native

import os
import re
import scf_guess_tools.engine as e


class Molecule(Base):
    @property
    def native(self) -> Native:
        return self._native

    @property
    def name(self) -> str:
        return self._native.name()

    @property
    def charge(self) -> int:
        return self._native.molecular_charge()

    @property
    def multiplicity(self) -> int:
        return self._native.multiplicity()

    @property
    def atoms(self) -> int:
        return self._native.natom()

    @property
    def geometry(self):
        return self._native.to_string(dtype="psi4")

    def __init__(self, engine: e.Engine, native: Native):
        super().__init__(engine)
        self._native = native

    def __getstate__(self):
        return self.name, self.geometry

    def __setstate__(self, serialized):
        self._native = Native.from_string(
            serialized[1], name=serialized[0], dtype="psi4"
        )

    @classmethod
    def load(cls, engine: e.Engine, path: str) -> Molecule:
        with open(path, "r") as file:
            lines = file.readlines()

        q = re.search(r"charge\s+(-?\d+)", lines[1]).group(1)
        m = re.search(r"multiplicity\s+(\d+)", lines[1]).group(1)

        lines[1] = f"{q} {m}\n"
        xyz = "".join(lines)

        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        return Molecule(engine, Native.from_string(xyz, name=name, dtype="xyz+"))
