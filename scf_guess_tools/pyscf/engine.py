from ..engine import Engine as Base


class Engine(Base):
    def __init__(self, cache: bool = True, verbose: int = 0):
        super().__init__("pyscf" if cache else None, verbose)

    @classmethod
    def __repr__(cls) -> str:
        return "PySCFEngine"

    @classmethod
    def guessing_schemes(cls) -> list[str]:
        return ["minao", "1e", "atom", "huckel", "vsap"]
