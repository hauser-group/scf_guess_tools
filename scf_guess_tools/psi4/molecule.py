import os
import re

from ..molecule import Molecule as Base
from psi4.core import Molecule as Native


class Molecule(Base):
    def __init__(self, path: str):
        with open(path, "r") as file:
            lines = file.readlines()

        q = re.search(r"charge\s+(-?\d+)", lines[1]).group(1)
        m = re.search(r"multiplicity\s+(\d+)", lines[1]).group(1)

        lines[1] = f"{q} {m}\n"
        xyz = "".join(lines)

        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        self._native = Native.from_string(xyz, name=name, dtype="xyz+")

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

    def __getstate__(self):
        return self.native.to_string(dtype="psi4"), self.name

    def __setstate__(self, serialized):
        self._native = Native.from_string(
            serialized[0], name=serialized[1], dtype="psi4"
        )
