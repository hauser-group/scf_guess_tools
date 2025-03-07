from contextlib import contextmanager

import psi4


@contextmanager
def clean_context():
    with psi4.driver.p4util.hold_options_state():
        psi4.core.clean_options()
        psi4.core.clean()

        yield

        psi4.core.clean_options()
        psi4.core.clean()
