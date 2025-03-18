from contextlib import contextmanager

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

    if exception is not None:
        raise exception
