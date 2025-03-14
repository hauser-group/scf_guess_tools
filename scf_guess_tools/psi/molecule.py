from __future__ import annotations

from ..molecule import Molecule as Base
from .core import Object
from psi4.core import Molecule as Native
from typing import overload

import os
import re


class Molecule(Base, Object):
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

    def __init__(self, native: Native):
        self._native = native

    def __getstate__(self):
        return super().__getstate__(), self.name, self.geometry

    def __setstate__(self, serialized):
        super().__setstate__(serialized[0])

        self._native = Native.from_string(
            serialized[2], name=serialized[1], dtype="psi4"
        )

    @classmethod
    def load(cls, path: str) -> Molecule:
        with open(path, "r") as file:
            lines = file.readlines()

        q = re.search(r"charge\s+(-?\d+)", lines[1]).group(1)
        m = re.search(r"multiplicity\s+(\d+)", lines[1]).group(1)

        lines[1] = f"{q} {m}\n"
        xyz = "".join(lines)

        base_name = os.path.basename(path)
        name, _ = os.path.splitext(base_name)

        return Molecule(Native.from_string(xyz, name=name, dtype="xyz+"))
