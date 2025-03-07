from __future__ import annotations

from abc import ABCMeta
from typing import Any


def tuplify(object: Any) -> tuple:
    return object if isinstance(object, tuple) else (object,)


def untuplify(object: tuple) -> Any | tuple:
    return object[0] if len(object) == 1 else object


class Singleton(ABCMeta, type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        elif kwargs.get("reinit_singleton", True):
            cls._instances[cls].__init__(*args, **kwargs)
        return cls._instances[cls]
