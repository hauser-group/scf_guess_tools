import os
import re

from psi4.core import Molecule


def load(path: str, disable_symmetry: bool = False) -> Molecule:
    with open(path, "r") as file:
        lines = file.readlines()

    q = re.search(r"charge\s+(-?\d+)", lines[1]).group(1)
    s = re.search(r"multiplicity\s+(\d+)", lines[1]).group(1)

    lines[1] = f"{q} {s}\n"
    xyz = "".join(lines)

    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)

    molecule = Molecule.from_string(xyz, name=file_name, dtype="xyz+")

    if disable_symmetry:
        molecule.reset_point_group("C1")

    return molecule


def charged(molecule: Molecule) -> bool:
    return molecule.molecular_charge() != 0


def non_charged(molecule: Molecule) -> bool:
    return not charged(molecule)


def singlet(molecule: Molecule) -> bool:
    return molecule.multiplicity() == 1


def non_singlet(molecule: Molecule) -> bool:
    return not singlet(molecule)
