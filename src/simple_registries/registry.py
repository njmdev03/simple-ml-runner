from typing import Dict, Any, Union
from inspect import isclass


class Registry:
    """Object that stores a unique dictionary of registered classes/methods/objects based on keys.

    Includes a decorator that will register wrapped methods/classes automatically.
    """
    def __init__(self, instantiate_classes: bool = False):
        """Create a new Registry object

        Args:
            instantiate_classes (bool, optional): If item getting registered is a class, instantiate it before
            registration. This can help create a singleton or global pattern if needed. Defaults to False.
        """
        self._instantiate_classes: bool = instantiate_classes
        self._registry: Dict[str, Any] = {}

    def register(self, item: Any, *keys: str, overwrite: bool = False) -> None:
        """Method to register a parser for given file extensions.

        Args:
            item: The item to register. If this item is a class and instantiate_classes was set during construction,
            then the item will be instantiated before registration.
            *keys: One or more keys to store the item under.
            overwrite: If there is already an item registered under a given key, should it be overwritten? Defaults to False.
        """
        if isclass(item) and self._instantiate_classes:
            item = item()

        for key in keys:
            if overwrite or self._registry.get(key) == None:
                self._registry[key] = item

    def get(self, key: str) -> Union[Any, None]:
        """Get a registered item by its key.

        Args:
            key: The key to search for.

        Returns:
            The registered item, or None if no item is registered under the given key.
        """
        return self._registry.get(key)

    def clear(self) -> None:
        """Remove all registered items from the registry.
        """
        self._registry.clear()

    def all(self) -> list:
        """Get all of the registered items. The returned list may contain
        duplicates if an item is registered under multiple keys.

        Returns:
            list: All of the registered items, including duplicates.
        """
        return self._registry.values()

    def keys(self) -> list:
        """Get all of the registered keys.

        Returns:
            list: All of the registered keys.
        """
        return self._registry.keys()

    def decorator(self, *keys) -> callable:
        """Get a decorator function that handles registering the wrapped
        method/class under the given keys.

        Returns:
            callable: The decorator method to wrap the method/class in.
        """
        def decorator(item):
            self.register(item, *keys)
            return item

        return decorator
