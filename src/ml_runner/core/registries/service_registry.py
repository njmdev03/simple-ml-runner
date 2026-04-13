from typing import Dict, Any, Type, TypeVar, Optional

T = TypeVar("T")

class ServiceRegistry:
    """
    A simple service registry for extension-to-extension communication.
    Allows registering and retrieving singleton service instances by type or name.
    """

    def __init__(self):
        self._services: Dict[str, Any] = {}
        self._type_map: Dict[Type, Any] = {}

    def register(self, service: Any, name: Optional[str] = None):
        """
        Registers a service instance.

        Args:
            service: The service instance to register.
            name: Optional name for the service. If not provided, the class name is used.
        """
        service_name = name or service.__class__.__name__
        self._services[service_name] = service
        self._type_map[type(service)] = service

        # Also register for all base classes (except object)
        for base in service.__class__.__mro__:
            if base is not object:
                self._type_map[base] = service

    def get(self, service_type_or_name: Any) -> Any:
        """
        Retrieves a registered service instance.

        Args:
            service_type_or_name: Either the type (class) of the service or its registered name.

        Returns:
            The service instance.

        Raises:
            KeyError: If the service is not found.
        """
        if isinstance(service_type_or_name, str):
            if service_type_or_name not in self._services:
                raise KeyError(f"Service '{service_type_or_name}' not found in registry.")
            return self._services[service_type_or_name]

        if service_type_or_name not in self._type_map:
            raise KeyError(f"Service of type '{service_type_or_name}' not found in registry.")
        return self._type_map[service_type_or_name]

    def has(self, service_type_or_name: Any) -> bool:
        """
        Checks if a service is registered.

        Args:
            service_type_or_name: Either the type (class) of the service or its registered name.

        Returns:
            True if the service exists, False otherwise.
        """
        if isinstance(service_type_or_name, str):
            return service_type_or_name in self._services
        return service_type_or_name in self._type_map
