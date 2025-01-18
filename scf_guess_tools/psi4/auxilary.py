import psi4

from contextlib import contextmanager
from tempfile import TemporaryDirectory


@contextmanager
def clean_context():
    with psi4.driver.p4util.hold_options_state():
        with TemporaryDirectory() as tmp:
            try:
                psi4.core.clean_options()
                psi4.core.clean()

                stdout_file = f"{tmp}/stdout"  # don't rename, bug with READ option
                psi4.extras.set_output_file(stdout_file)

                yield stdout_file
            finally:
                psi4.core.clean_options()
                psi4.core.clean()
                psi4.core.set_output_file("/dev/stdout", False)
