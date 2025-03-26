from __future__ import annotations

from .core import Object, cache_directory
from functools import wraps
from pathlib import Path
from time import process_time
from typing import Any, Callable

import inspect
import joblib
import logging
import pickle


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


def cache(
    directory: str | None = None, ignore: list[str] | None = None, enable: bool = True
):
    """Store calculated functions in a cache directory and reload them if the function
    is invoked with the same set of parameters.

    Args:
        directory: The cache directory to store the calculated results in. If None, use
            the SGT_CACHE_DIR environment variable.
        ignore: A list of functions arguments (e.g. cls, self, verbose) to exclude from
            the hash key.
        enable: Whether caching is enabled by default.

    Returns:
        The wrapped function with caching functionality.
    """
    ignore = ignore or []
    logger = logging.getLogger("CACHE")

    def decorator(function):
        @wraps(function)
        def wrapper(*args, cache: bool | str = enable, **kwargs):
            if cache == False:
                return function(*args, **kwargs)
            elif cache == True:
                cache = directory

            cache = cache or cache_directory(throw=True)
            name = f"{function.__module__}.{function.__qualname__}"
            base = Path(cache) / name

            signature = inspect.signature(function)
            arguments = signature.bind(*args, **kwargs)
            arguments.apply_defaults()

            def canonicalize(object):
                if isinstance(object, (list, tuple)):
                    return tuple(canonicalize(e) for e in object)
                elif isinstance(object, dict):
                    return {k: canonicalize(object[k]) for k in sorted(object.keys())}
                elif isinstance(object, Object):
                    return object.__hash__()
                elif hasattr(object, "__dict__"):
                    return canonicalize(vars(object))
                elif isinstance(object, (str, int, float, complex, bool, type(None))):
                    return object
                else:
                    raise TypeError(f"Unable to canonicalize {object}")

            parameters = {
                k: canonicalize(v)
                for k, v in arguments.arguments.items()
                if k not in ignore
            }

            key = joblib.hash(parameters)
            result_available, result = None, None
            bucket = 1

            while result_available is None:
                parameters_file = base / f"{key}.parameters.{bucket}.pkl"
                result_file = base / f"{key}.result.{bucket}.pkl"

                if not parameters_file.is_file() or not result_file.is_file():
                    result_available = False
                    break

                with parameters_file.open("rb") as file:
                    loaded_parameters = pickle.load(file)

                if loaded_parameters == parameters:
                    with result_file.open("rb") as file:
                        result = pickle.load(file)
                        result_available = True

                bucket += 1

            if result_available:
                logger.info(f"Returning cached result of {name}({arguments.arguments})")
                return result

            logger.info(f"Invoking {name}({arguments.arguments})")
            base.mkdir(parents=True, exist_ok=True)

            result = function(*args, **kwargs)
            bucket = 1

            while True:
                parameters_file = base / f"{key}.parameters.{bucket}.pkl"
                result_file = base / f"{key}.result.{bucket}.pkl"

                if not parameters_file.is_file():
                    with parameters_file.open("wb") as file:
                        pickle.dump(parameters, file)

                    with result_file.open("wb") as file:
                        pickle.dump(result, file)

                    break

                with parameters_file.open("rb") as file:
                    loaded_parameters = pickle.load(file)

                assert loaded_parameters != parameters

                logger.info(f"Hash collision detected for bucket {bucket}")
                bucket += 1

            return result

        return wrapper

    return decorator
