from ..core import Backend, Object as Base, cache_directory

import os
import psi4


class Object(Base):
    @classmethod
    def backend(cls) -> Backend:
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

if not cache_directory:
    raise RuntimeError("SGT_CACHE environment variable not set")

output_directory = f"{cache_directory}/{Backend.PSI.value}"
os.makedirs(output_directory, exist_ok=True)

output_file = f"{output_directory}/stdout"
psi4.set_output_file(output_file, append=False, execute=True, print_header=False)
