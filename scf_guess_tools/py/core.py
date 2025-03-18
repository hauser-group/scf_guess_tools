from ..core import Backend, Object as Base


class Object(Base):
    """Base class for objects that use the PySCF backend."""

    @classmethod
    def backend(cls) -> Backend:
        """The backend associated with this object."""
        return Backend.PY


guessing_schemes = ["minao", "1e", "atom", "huckel", "vsap"]


def reset():
    """Reset the PySCF backend state. Currently, this function does nothing."""
    pass
