from typing import Callable, Dict


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
    def all(cls):
        return list(cls._registry.keys())
