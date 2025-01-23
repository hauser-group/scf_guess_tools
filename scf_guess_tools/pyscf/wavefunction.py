from ..wavefunction import Wavefunction as Base
from .molecule import Molecule
from pyscf.scf import RHF, UHF
from typing import Self
from numpy.typing import NDArray


class Wavefunction(Base):
    def __init__(self, native: NDArray, molecule: Molecule, origin: str):
        super().__init__(molecule, origin)
        self._native = native

        if molecule.singlet:
            self._Da = self._Db = native / 2
        else:
            self._Da, self._Db = native

    @classmethod
    def guess(cls, molecule: Molecule, basis: str, method: str) -> Self:
        molecule.native.basis = basis
        molecule.native.build()

        guesser = RHF if molecule.singlet else UHF
        guess = guesser(molecule.native).get_init_guess(molecule.native, method)

        return Wavefunction(guess, molecule, method)

    @property
    def native(self) -> NDArray:
        return self._native

    @property
    def Da(self) -> NDArray:
        return self._Da

    @property
    def Db(self) -> NDArray:
        return self._Db
