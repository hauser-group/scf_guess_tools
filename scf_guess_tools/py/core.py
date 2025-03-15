from ..core import Backend, Object as Base


class Object(Base):
    @classmethod
    def backend(cls) -> Backend:
        return Backend.PY


guessing_schemes = ["minao", "1e", "atom", "huckel", "vsap"]


def reset():
    pass
