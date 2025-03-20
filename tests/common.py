from __future__ import annotations

from scf_guess_tools import Matrix, Molecule, Wavefunction
from scf_guess_tools.common import tuplify
from scf_guess_tools.psi.matrix import Matrix as PsiMatrix
from scf_guess_tools.psi.molecule import Molecule as PsiMolecule
from scf_guess_tools.psi.wavefunction import Wavefunction as PsiWavefunction


import numpy as np
import random
import re


def replace_random_digit(original: str, modified: str):
    with open(original, "r", encoding="utf-8") as file:
        header = file.readline()
        header += file.readline()
        content = file.read()

    positions = [match.start() for match in re.finditer(r"\d", content)]
    assert positions, "file should contain digits"

    index = random.choice(positions)
    old = content[index]
    new = random.choice([digit for digit in "0123456789" if digit != old])

    content = content[:index] + new + content[index + 1 :]

    with open(modified, "w", encoding="utf-8") as file:
        file.write(header)
        file.write(content)


def equal_matrices(a: Matrix, b: Matrix) -> bool:
    assert a.backend() == b.backend()

    if isinstance(a, PsiMatrix) and isinstance(b, PsiMatrix):
        for lhs, rhs in zip(a.native.nph, b.native.nph):
            if not np.allclose(lhs, rhs, rtol=1e-6, atol=1e-6):
                return False

        return True

    return np.array_equal(a.native, b.native)


def equal_molecules(a: Molecule, b: Molecule, ignore: list[str] | None = None) -> bool:
    assert a.backend() == b.backend()
    ignore = ignore or []

    properties = [
        "name",
        "charge",
        "multiplicity",
        "singlet",
        "atoms",
        "geometry",
    ]

    for property in properties:
        if property in ignore:
            continue

        if getattr(a, property) != getattr(b, property):
            print(f"Molecule.{property} differs for {a} and {b}")
            return False

    return True


def equal_wavefunctions(
    a: Wavefunction, b: Wavefunction, ignore: list[str] | None = None
) -> bool:
    assert a.backend() == b.backend()
    ignore = ignore or []

    properties = [
        "basis",
        "origin",
        "time",
        "converged",
        "stable",
        "second_order",
    ]

    if not "molecule" in ignore and not equal_molecules(a.molecule, b.molecule):
        print(f"Wavefunction.molecule differs for {a} and {b}")
        return False

    if not "initial" in ignore and not equal(a.initial, b.initial):
        print(f"Wavefunction.initial differs for {a} and {b}")
        return False

    for property in properties:
        if property in ignore:
            continue

        if getattr(a, property) != getattr(b, property):
            print(f"Wavefunction.{property} differs for {a} and {b}")
            return False

    for property in ["overlap", "density", "fock", "core_hamiltonian"]:
        if property in ignore:
            continue

        lhs, rhs = tuplify(getattr(a, property)()), tuplify(getattr(b, property)())

        for x, y in zip(lhs, rhs):
            if not equal_matrices(x, y):
                print(f"Wavefunction.{property} differs for {a} and {b}")
                return False

    if not "electronic_energy" in ignore and not np.isclose(
        a.electronic_energy(), b.electronic_energy(), rtol=1e-5, atol=1e-10
    ):
        print(
            f"Wavefunction.electronic_energy differs for {a} and {b} by {a.electronic_energy() - b.electronic_energy()}"
        )
        return False

    return True


def equal(a, b, **kwargs) -> bool:
    if isinstance(a, Matrix) and isinstance(b, Matrix):
        return equal_matrices(a, b, **kwargs)
    elif isinstance(a, Molecule) and isinstance(b, Molecule):
        return equal_molecules(a, b, **kwargs)
    elif isinstance(a, Wavefunction) and isinstance(b, Wavefunction):
        return equal_wavefunctions(a, b, **kwargs)

    return a == b


def similar_matrices(
    a: Matrix | tuple[Matrix, Matrix],
    b: Matrix | tuple[Matrix, Matrix],
    tolerance: float,
) -> True:
    a, b = tuplify(a), tuplify(b)

    for lhs, rhs in zip(a, b):
        trace = abs(lhs.trace / rhs.trace - 1.0)
        if trace > tolerance:
            print("not similar trace ", trace)
            return False

        purity = abs((lhs @ lhs).trace / (rhs @ rhs).trace - 1.0)
        if purity > tolerance:
            print("not similar purity ", purity)
            return False

    return True


def similar_wavefunctions(
    a: Wavefunction, b: Wavefunction, tolerance: float, ignore: list[str]
) -> bool:
    if not "molecule" in ignore and not equal_molecules(a.molecule, b.molecule):
        print(f"Wavefunction.molecule differs for {a} and {b}")
        return False

    properties = [
        "basis",
        "origin",
        "time",
        "converged",
        "stable",
        "second_order",
    ]

    for property in properties:
        if property in ignore:
            continue

        if getattr(a, property) != getattr(b, property):
            print(f"Wavefunction.{property} differs for {a} and {b}")
            return False

    if not "initial" in ignore and not similar(
        a.initial, b.initial, tolerance, ignore=ignore
    ):
        print(f"Wavefunction.initial not similar for {a} and {b}")
        return False

    for property in ["overlap", "density", "fock", "core_hamiltonian"]:
        if property in ignore:
            continue

        lhs, rhs = tuplify(getattr(a, property)()), tuplify(getattr(b, property)())

        for x, y in zip(lhs, rhs):
            if not similar_matrices(x, y, tolerance * 1e3):
                print(f"Wavefunction.{property} not similar for {a} and {b}")
                return False

    if not "electronic_energy" in ignore and not similar(
        a.electronic_energy(), b.electronic_energy(), tolerance * 1e3
    ):
        print(f"Wavefunction.electronic_energy not similar for {a} and {b}")
        return False

    return True


def similar(a, b, tolerance: float = 1e-5, ignore: list[str] | None = None) -> bool:
    ignore = ignore or []

    if isinstance(a, Matrix) and isinstance(b, Matrix):
        return similar_matrices(a, b, tolerance, ignore=ignore)
    elif isinstance(a, Wavefunction) and isinstance(b, Wavefunction):
        return similar_wavefunctions(a, b, tolerance, ignore=ignore)
    elif isinstance(a, str) and isinstance(b, str):
        return a == b

    return abs(a / b - 1.0) < tolerance
