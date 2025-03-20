from .core import reset
from contextlib import contextmanager
from .core import output_directory, output_file
import os

import psi4


@contextmanager
def clean_context():
    """Context manager that provides a clean Psi4 configuration. This ensures that
    Psi4's options, internal state and temporary files are reset before executing the
    enclosed code block. The previous set of options is restored afterwards.
    """
    exception = None
    reset()

    with psi4.driver.p4util.hold_options_state():
        psi4.core.clean_options()
        psi4.core.clean()

        try:
            yield
        except Exception as e:
            exception = e

        # leave files for debugging

    if exception is not None:
        raise exception
