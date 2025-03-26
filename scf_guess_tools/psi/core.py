from ..core import Backend, Object as Base
from pathlib import Path

import os
import psi4
import shutil


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

# The following code must only be called once
# Psi4 does not close filehandles again, and the process will eventually be killed for
# opening too many files

output_directory: str = os.environ.get("PSI_SCRATCH")

if output_directory is None:
    raise RuntimeError("PSI_SCRATCH environment variable is not set")

output_file = f"{output_directory}/stdout"
log_file = f"{output_directory}/stdout.log"

psi4.set_output_file(output_file, append=False, execute=True, print_header=False)


def reset():
    """Reset the Psi4 backend state. Currently, this function resets the scratch
    directory and cleans output files."""

    base = Path(output_directory)
    keep = [Path(output_file), Path(log_file)]

    for item in base.iterdir():
        if item in keep:
            continue

        if item.is_dir():
            shutil.rmtree(item)
        elif item.is_file():
            item.unlink()

    for path in keep:
        with open(path, "w+") as file:
            file.truncate(0)
