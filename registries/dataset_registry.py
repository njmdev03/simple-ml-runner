class DatasetRegistry:
    _datasets = {}

    @classmethod
    def register(cls, name: str):
        """
        Decorator to register a dataset class under a string key.
        """
        def decorator(fn_or_cls):
            cls._datasets[name] = fn_or_cls
            return fn_or_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._datasets:
            raise ValueError(f"No dataset registered under name: {name}")
        return cls._datasets[name]

    @classmethod
    def all(cls):
        return cls._datasets.keys()