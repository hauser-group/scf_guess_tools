from .matrix import Matrix
from .molecule import Molecule
from ..wavefunction import Wavefunction as Base
from pyscf.scf import RHF, UHF
from pyscf.scf.hf import SCF
from typing import Self
from numpy.typing import NDArray


class Wavefunction(Base):

    def __init__(self, solver: SCF, D: NDArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._solver = solver
        self._D = D

    @property
    def native(self) -> SCF:
        return self._solver

    @property
    def S(self) -> Matrix:
        return Matrix(self._solver.get_ovlp())

    @property
    def D(self) -> Matrix | tuple[Matrix, Matrix]:
        if self.molecule.singlet:
            return Matrix(self._D / 2)

        return Matrix(self._D[0]), Matrix(self._D[1])

    @property
    def F(self) -> Matrix | tuple[Matrix, Matrix]:
        F = self._solver.get_fock(dm=self._D)

        if self.molecule.singlet:
            return Matrix(F)

        return Matrix(F[0]), Matrix(F[1])

    @classmethod
    def guess(cls, molecule: Molecule, basis: str, scheme: str) -> Self:
        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        solver = method(molecule.native)

        D = solver.get_init_guess(key=scheme)

        return Wavefunction(solver, D, molecule, basis, scheme)

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Self = None
    ) -> Self:
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
            solver, solver.make_rdm1(), molecule, basis, guess, solver.cycles, retry
        )

    def __getstate__(self):
        return super().__getstate__(), self._D

    def __setstate__(self, serialized):
        super().__setstate__(serialized[0])
        self._D = serialized[1]

        self._molecule.native.basis = self._basis
        self._molecule.native.build()

        method = RHF if self._molecule.singlet else UHF
        self._solver = method(self._molecule.native)
