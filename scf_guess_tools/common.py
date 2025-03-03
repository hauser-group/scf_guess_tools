def tuplify(object) -> tuple:
    return object if isinstance(object, tuple) else (object,)
