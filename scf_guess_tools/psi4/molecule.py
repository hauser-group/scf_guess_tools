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
        self._name, _ = os.path.splitext(base_name)

        self._molecule = Native.from_string(xyz, name=self._name,
                                            dtype="xyz+")

    @property
    def name(self) -> str:
        return self._name

    @property
    def charge(self) -> int:
        return self._molecule.molecular_charge()

    @property
    def multiplicity(self) -> int:
        return self._molecule.multiplicity()

    @property
    def atoms(self) -> int:
        return self._molecule.natom()

    @property
    def native(self) -> Native:
        return self._molecule
