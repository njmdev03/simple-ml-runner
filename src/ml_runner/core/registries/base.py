from typing import Any, Callable, Dict


class BaseRegistry:
    """
    Generic registry mapping string names -> callables/classes.
    """
    _registry: Dict[str, Callable] = {}

    @classmethod
    def register(cls, *names: str):
        def decorator(obj: Callable):
            for name in names:
                key = name.lower()

                if key in cls._registry:
                    raise ValueError(
                        f"{cls.__name__}: '{name}' already registered"
                    )

                cls._registry[key] = obj
            return obj

        return decorator

    @classmethod
    def get(cls, name: str) -> Callable:
        key = name.lower()

        if key not in cls._registry:
            raise ValueError(
                f"{cls.__name__}: no component registered under '{name}'. "
                f"Available: {list(cls._registry.keys())}"
            )

        return cls._registry[key]

    @classmethod
    def get_name(cls, obj: Any) -> str:
        """
        Reverse lookup to find the first registered name for an object.
        """
        for name, registered_obj in cls._registry.items():
            if registered_obj == obj or (hasattr(obj, '__class__') and registered_obj == obj.__class__):
                return name
        return getattr(obj, "__name__", obj.__class__.__name__)

    @classmethod
    def contains(cls, obj: Any) -> bool:
        return any(v == obj for v in cls._registry.values())

    @classmethod
    def all(cls):
        return list(cls._registry.keys())
