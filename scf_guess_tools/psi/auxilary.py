from contextlib import contextmanager
from .core import output_directory, output_file
import os

import psi4


@contextmanager
def clean_context():
    """Context manager that provides a clean Psi4 configuration. This ensures that
    Psi4's options and internal state are reset before executing the enclosed code block.
    The previous set of options is restored afterwards.
    """
    exception = None

    with psi4.driver.p4util.hold_options_state():
        psi4.core.clean_options()
        psi4.core.clean()

        try:
            yield
        except Exception as e:
            exception = e
        finally:
            psi4.core.clean_options()
            psi4.core.clean()
            if os.path.exists(output_file):
                os.remove(output_file)
                psi4.set_output_file(
                    output_file, append=False, execute=True, print_header=False
                )

    if exception is not None:
        raise exception
