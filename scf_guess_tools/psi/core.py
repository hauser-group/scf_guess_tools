from ..core import Backend, Object as Base, cache_directory

import os
import psi4


class Object(Base):
    """Base class for objects that use the Psi4 backend."""

    @classmethod
    def backend(cls) -> Backend:
        """The backend associated with this object."""
        return Backend.PSI


guessing_schemes = [
    "CORE",
    "SAD",
    "SADNO",
    "GWH",
    "HUCKEL",
    "MODHUCKEL",
    "SAP",
    "SAPGAU",
]

output_directory: str = None
output_file: str = None


def reset():
    """Reset the Psi4 backend output configuration. Ensures that the output directory
    exists and sets the output file for Psi4.
    """
    global output_directory
    global output_file

    output_directory = f"{cache_directory(throw=True)}.psi"  # don't place within
    os.makedirs(output_directory, exist_ok=True)

    output_file = f"{output_directory}/stdout"
    psi4.set_output_file(output_file, append=False, execute=True, print_header=False)


reset()
