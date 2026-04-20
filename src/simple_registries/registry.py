from typing import Dict, Any, Union, List
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

    def register(self, item: Any, *keys: Any, overwrite: bool = False) -> Union[List[Any], None]:
        """Method to register a parser for given file extensions.

        Args:
            item: The item to register. If this item is a class and instantiate_classes was set during construction,
            then the item will be instantiated before registration.
            *keys: One or more keys to store the item under.
            overwrite: If there is already an item registered under a given key, should it be overwritten? Defaults to
            False.

        Returns:
            Union[Any, None]: Returns the list of keys the item was registered under, or None if the item was not
            registered.
        """
        if isclass(item) and self._instantiate_classes:
            item = item()

        reg_keys = []

        for key in keys:
            if overwrite or self._registry.get(key) == None:
                self._registry[key] = item
                reg_keys.append(key)

        return None if reg_keys == [] else reg_keys

    def get(self, key: Any) -> Union[Any, None]:
        """Get a registered item by its key.

        Args:
            key: The key to search for.

        Returns:
            The registered item, or None if no item is registered under the given key.
        """
        return self._registry.get(key)

    def remove(self, key: Any) -> Union[Any, None]:
        """Remove the item registered at the given key.

        Args:
            key (Any): The key to delete the entry of

        Returns:
            Union[Any, None]: Return the deleted value if something was removed, otherwise None if nothing was removed.
        """
        return self._registry.pop(key, None)

    def deregister(self, item: Any) -> Union[List[Any], None]:
        """Removed an item from the registry at all of the keys it was registered to. This is an expensive operation
        since a reverse search/lookup must be performed for each item removed.

        Args:
            item (Any): The item to remove from the registry

        Returns:
            Union[List[Any], None]: None if no instances of the item were found, otherwise a list of keys that the item
            was registered under.
        """
        removed_keys = []

        for k, v in list(self._registry.items()):
            if v == item:
                removed_keys.append(k)
                self._registry.pop(k)

        return None if removed_keys == [] else removed_keys

    def clear(self) -> None:
        """Remove all registered items from the registry.
        """
        self._registry.clear()

    def all(self) -> List:
        """Get all of the registered items. The returned list may contain
        duplicates if an item is registered under multiple keys.

        Returns:
            list: All of the registered items, including duplicates.
        """
        return self._registry.values()

    def keys(self) -> List:
        """Get all of the registered keys.

        Returns:
            list: All of the registered keys.
        """
        return self._registry.keys()
