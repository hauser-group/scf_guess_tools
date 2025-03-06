from __future__ import annotations

from ..wavefunction import Wavefunction as Base
from .matrix import Matrix
from numpy.typing import NDArray
from pyscf.scf import RHF, UHF
from pyscf.scf.hf import SCF as Native

import scf_guess_tools.pyscf.engine as e
import scf_guess_tools.pyscf.molecule as m


class Wavefunction(Base):
    @property
    def native(self) -> Native:
        return self._native

    @property
    def S(self) -> Matrix:
        return Matrix(self._native.get_ovlp())

    @property
    def H(self) -> Matrix:
        return Matrix(self.native.get_hcore())

    @property
    def D(self) -> Matrix | tuple[Matrix, Matrix]:
        if self.molecule.singlet:
            return Matrix(self._D / 2)

        return Matrix(self._D[0]), Matrix(self._D[1])

    @property
    def F(self) -> Matrix | tuple[Matrix, Matrix]:
        F = self._native.get_fock(dm=self._D)

        if self.molecule.singlet:
            return Matrix(F)

        return Matrix(F[0]), Matrix(F[1])

    def __init__(self, native: Native, D: NDArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native = native
        self._D = D

    def __getstate__(self):
        return super().__getstate__(), self._D

    def __setstate__(self, serialized):
        super().__setstate__(serialized[0])
        self._D = serialized[1]

        self._molecule.native.basis = self._basis
        self._molecule.native.build()

        method = RHF if self._molecule.singlet else UHF
        self._native = method(self._molecule.native)

    @classmethod
    def guess(
        cls, engine: e.Engine, molecule: m.Molecule, basis: str, scheme: str
    ) -> Wavefunction:
        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        solver = method(molecule.native)

        D = solver.get_init_guess(key=scheme)

        return Wavefunction(solver, D, engine, molecule, basis, scheme, converged=False)

    @classmethod
    def calculate(
        cls,
        engine: e.Engine,
        molecule: m.Molecule,
        basis: str,
        guess: str | Wavefunction | None = None,
    ) -> Wavefunction:
        guess = "minao" if guess is None else guess

        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        solver = method(molecule.native)

        if isinstance(guess, str):
            solver.init_guess = guess
        else:
            solver.init_guess = guess.native

        solver.run()
        retry = False

        if molecule.atoms <= 30:
            mo, _, stable, _ = solver.stability(return_status=True)
            retry = not stable

            while not stable:
                dm = solver.make_rdm1(mo, solver.mo_occ)
                solver = solver.run(dm)
                mo, _, stable, _ = solver.stability(return_status=True)

        return Wavefunction(
            solver,
            solver.make_rdm1(),
            engine,
            molecule,
            basis,
            guess,
            solver.cycles,
            retry,
            solver.converged,
        )
