from __future__ import annotations

from ..builder import builder_property
from ..wavefunction import Wavefunction as Base
from .engine import Engine
from .matrix import Matrix
from .molecule import Molecule
from numpy.typing import NDArray
from pyscf.scf import RHF, UHF
from pyscf.scf.hf import SCF as Native
from time import process_time


class Wavefunction(Base):
    @property
    def native(self) -> Native:
        return self._native

    @builder_property
    def S(self) -> Matrix:
        return Matrix(self._native.get_ovlp())

    @builder_property
    def H(self) -> Matrix:
        return Matrix(self.native.get_hcore())

    @builder_property
    def D(self) -> Matrix | tuple[Matrix, Matrix]:
        if self.molecule.singlet:
            return Matrix(self._D / 2)

        return Matrix(self._D[0]), Matrix(self._D[1])

    @builder_property
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
    def engine(cls) -> Engine:
        return Engine(reinit_singleton=False)

    @classmethod
    def guess(
        cls, molecule: Molecule, basis: str, scheme: str | None = None
    ) -> Wavefunction:
        start = process_time()

        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        solver = method(molecule.native)

        scheme = solver.init_guess if scheme is None else scheme

        D = solver.get_init_guess(key=scheme)
        end = process_time()

        return Wavefunction(
            solver,
            D,
            molecule=molecule,
            basis=basis,
            initial=scheme,
            origin="guess",
            time=end - start,
        )

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Wavefunction | None = None
    ) -> Wavefunction:
        start = process_time()

        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        solver = method(molecule.native)

        guess = solver.init_guess if guess is None else guess

        if isinstance(guess, str):
            assert guess in Engine(reinit_singleton=False).guessing_schemes()
            solver.run(init_guess=guess)
        else:
            solver.kernel(dm0=guess._D)

        retry = False

        if molecule.atoms <= 30:
            mo, _, stable, _ = solver.stability(return_status=True)
            retry = not stable

            while not stable:
                dm = solver.make_rdm1(mo, solver.mo_occ)
                solver = solver.run(dm)
                mo, _, stable, _ = solver.stability(return_status=True)

        end = process_time()

        return Wavefunction(
            solver,
            solver.make_rdm1(),
            molecule=molecule,
            basis=basis,
            initial=guess,
            origin="calculation",
            time=end - start,
            iterations=solver.cycles,
            retried=retry,
            converged=solver.converged,
        )
