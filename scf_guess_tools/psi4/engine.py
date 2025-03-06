from ..engine import Engine as Base
from tempfile import TemporaryDirectory

import psi4


class Engine(Base):
    def __init__(self, cache: bool = True, verbose: int = 0):
        super().__init__("psi4" if cache else None, verbose)

        self._output_directory = TemporaryDirectory()

        psi4.set_output_file(
            self.output_file, append=False, execute=True, print_header=False
        )

    @property
    def output_file(self) -> str:
        return f"{self._output_directory.name}/stdout"  # don't rename, bug with READ option

    @classmethod
    def __repr__(cls) -> str:
        return "Psi4Engine"

    @classmethod
    def guessing_schemes(cls) -> list[str]:
        return ["CORE", "SAD", "SADNO", "GWH", "HUCKEL", "MODHUCKEL", "SAP", "SAPGAU"]
