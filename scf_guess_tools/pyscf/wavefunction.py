from ..wavefunction import Wavefunction as Base
from .molecule import Molecule
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
    def native(self) -> NDArray:
        return self._D

    @property
    def S(self) -> NDArray:
        return self._solver.get_ovlp()

    @property
    def D(self) -> NDArray:
        return self._D / 2 if self.molecule.singlet else self._D

    @property
    def F(self) -> NDArray:
        return self._solver.get_fock(dm=self._D)

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
        return (
            self.molecule,
            self.basis,
            self.initial,
            self.iterations,
            self.retried,
            self.native,
        )

    def __setstate__(self, serialized):
        self._molecule = serialized[0]
        self._basis = serialized[1]
        self._initial = serialized[2]
        self._iterations = serialized[3]
        self._retried = serialized[4]
        self._D = serialized[5]

        self._molecule.native.basis = self._basis
        self._molecule.native.build()

        method = RHF if self._molecule.singlet else UHF
        self._solver = method(self._molecule.native)
