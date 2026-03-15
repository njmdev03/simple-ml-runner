class ModelRegistry:
    _models = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a model class under a string key.
        """
        def decorator(fn_or_cls):
            cls._models[name] = fn_or_cls
            return fn_or_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._models:
            raise ValueError(f"No model registered under name: {name}")
        return cls._models[name]

    @classmethod
    def all(cls):
        return cls._models.keys()