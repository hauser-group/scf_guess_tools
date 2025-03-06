from typing import Any


def tuplify(object: Any) -> tuple:
    return object if isinstance(object, tuple) else (object,)


def untuplify(object: tuple) -> Any | tuple:
    return object[0] if len(object) == 1 else object
