from abc import ABC, abstractmethod
from typing import Any, Union, List


class AbstractClassRegistry(ABC):
    @classmethod
    @abstractmethod
    def __init__(cls, instantiate_classes: bool = False):
        return

    @classmethod
    @abstractmethod
    def register(cls, item: Any, *keys: str, overwrite: bool = False) -> Union[List[Any], None]:
        return

    @classmethod
    @abstractmethod
    def get(cls, key: str) -> Union[Any, None]:
        return

    @classmethod
    @abstractmethod
    def remove(cls, key: str) -> Union[Any, None]:
        return

    @classmethod
    @abstractmethod
    def deregister(cls, key: str) -> Union[List[Any], None]:
        return

    @classmethod
    @abstractmethod
    def clear(cls) -> None:
        return

    @classmethod
    @abstractmethod
    def all(cls) -> List:
        return

    @classmethod
    @abstractmethod
    def keys(cls) -> List:
        return

    # @classmethod
    # @abstractmethod
    # def decorator(cls, *keys) -> callable:
    #     return
