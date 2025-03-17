from __future__ import annotations

from functools import wraps
from time import process_time
from typing import Any, Callable


def tuplify(object: Any) -> tuple:
    return object if isinstance(object, tuple) else (object,)


def untuplify(object: tuple) -> Any | tuple:
    return object[0] if len(object) == 1 else object


def tuplifyable(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, tuplify: bool = False, **kwargs):
        result = function(*args, **kwargs)

        if tuplify and not isinstance(result, tuple):
            return (result,)

        return result

    return wrapper


def timeable(function: Callable) -> Callable:
    @wraps(function)
    def wrapper(*args, time: bool = False, **kwargs):
        start = process_time()
        result = function(*args, **kwargs)
        duration = process_time() - start

        return (result, duration) if time else result

    return wrapper
