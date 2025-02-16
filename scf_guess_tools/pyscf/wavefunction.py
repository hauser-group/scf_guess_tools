from ..wavefunction import Wavefunction as Base
from .molecule import Molecule
from pyscf.scf import RHF, UHF
from typing import Self
from numpy.typing import NDArray


class Wavefunction(Base):

    def __init__(self, native: NDArray, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._native = native

    @property
    def native(self) -> NDArray:
        return self._native

    @property
    def Da(self) -> NDArray:
        return self.native / 2 if self.molecule.singlet else self.native

    @property
    def Db(self) -> NDArray:
        return self.Da

    @classmethod
    def guess(cls, molecule: Molecule, basis: str, method: str) -> Self:
        molecule.native.basis = basis
        molecule.native.build()

        guesser = RHF if molecule.singlet else UHF
        guess = guesser(molecule.native).get_init_guess(molecule.native, method)

        return Wavefunction(guess, molecule, method)

    @classmethod
    def calculate(
        cls, molecule: Molecule, basis: str, guess: str | Self = None
    ) -> Self:
        guess = "minao" if guess is None else guess

        molecule.native.basis = basis
        molecule.native.build()

        method = RHF if molecule.singlet else UHF
        calculation = method(molecule.native)

        if isinstance(guess, str):
            calculation.init_guess = guess
        else:
            calculation.init_guess = guess.native

        calculation.run()
        retry = False

        if molecule.atoms <= 30:
            mo, _, stable, _ = calculation.stability(return_status=True)
            retry = not stable

            while not stable:
                dm = calculation.make_rdm1(mo, calculation.mo_occ)
                calculation = calculation.run(dm)
                mo, _, stable, _ = calculation.stability(return_status=True)

        return Wavefunction(
            calculation.make_rdm1(), molecule, guess, calculation.cycles, retry
        )

    def __getstate__(self):
        return (self.molecule, self.initial, self.iterations, self.retried, self.native)

    def __setstate__(self, serialized):
        self._molecule = serialized[0]
        self._initial = serialized[1]
        self._iterations = serialized[2]
        self._retried = serialized[3]
        self._native = serialized[4]
