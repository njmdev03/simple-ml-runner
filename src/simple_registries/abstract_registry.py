from abc import ABC, abstractmethod
from typing import Any, Union


class AbstractClassRegistry(ABC):
    @classmethod
    @abstractmethod
    def register(cls, item: Any, *keys: str, overwrite: bool = False) -> None:
        return

    @classmethod
    @abstractmethod
    def get(cls, key: str) -> Union[Any, None]:
        return

    @classmethod
    @abstractmethod
    def clear(cls) -> None:
        return

    @classmethod
    @abstractmethod
    def all(cls) -> list:
        return

    @classmethod
    @abstractmethod
    def keys(cls) -> list:
        return

    @classmethod
    @abstractmethod
    def decorator(cls, *keys) -> callable:
        return
