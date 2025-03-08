from abc import ABC, abstractmethod
from collections import defaultdict
from time import process_time
from typing import Type


class Builder(ABC):
    def __init__(self):
        self._builder_properties = defaultdict(lambda: defaultdict(lambda: None))

    def __getattr__(self, name: str):
        if not name.endswith("_cache") and not name.endswith("_time"):
            return super().__getattribute__(name)

        key, type = name.rsplit("_", 1)
        return self._builder_properties[key][type]

    @classmethod
    @abstractmethod
    def engine(cls) -> "Engine":
        pass


class BuilderProperty(property):
    def __init__(self, fget):
        super().__init__(fget)
        self._key = fget.__name__

    def __get__(self, instance: Builder, owner: Type[Builder]):
        if instance is None:
            return self

        if self._key in instance.engine().cached_properties:
            result = instance._builder_properties[self._key]["cache"]

            if result is not None:
                return result

        start_time = process_time()
        result = super().__get__(instance, owner)
        end_time = process_time()

        instance._builder_properties[self._key]["cache"] = result
        instance._builder_properties[self._key]["time"] = end_time - start_time

        return result


def builder_property(function) -> BuilderProperty:
    return BuilderProperty(function)
