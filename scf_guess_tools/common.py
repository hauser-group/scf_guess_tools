from __future__ import annotations

from functools import wraps
from time import process_time
from typing import Any, Callable


def tuplify(object: Any) -> tuple:
    """Ensure the object is a tuple.

    Args:
        object: The object to convert.

    Returns:
        A tuple containing the object if it is not already a tuple.
    """
    return object if isinstance(object, tuple) else (object,)


def untuplify(object: tuple) -> Any | tuple:
    """Unpack a tuple if it contains only one element.

    Args:
        object: The tuple to unpack.

    Returns:
        The single element if the tuple has only one item, otherwise the tuple itself.
    """
    return object[0] if len(object) == 1 else object


def tuplifyable(function: Callable) -> Callable:
    """Make a function return a tuple if requested. Functions decorated with this gain
    an additional tuplify parameter. If set to True, the return value is always a tuple.

    Args:
        function: The function to decorate.

    Returns:
        The wrapped function with tuplify functionality.
    """

    @wraps(function)
    def wrapper(*args, tuplify: bool = False, **kwargs):
        result = function(*args, **kwargs)

        if tuplify and not isinstance(result, tuple):
            return (result,)

        return result

    return wrapper


def timeable(function: Callable) -> Callable:
    """Measure execution time of a function. Functions decorated with this gain an
    additional time parameter. If set to True, the return value becomes a tuple (result,
    time), where time is the CPU time measured by time.process_time().

    Args:
        function: The function to decorate.

    Returns:
        The wrapped function with timing functionality.
    """

    @wraps(function)
    def wrapper(*args, time: bool = False, **kwargs):
        start = process_time()
        result = function(*args, **kwargs)
        duration = process_time() - start

        return (result, duration) if time else result

    return wrapper
