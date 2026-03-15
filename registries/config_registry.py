class ConfigRegistry:
    _parsers = {}

    @classmethod
    def register(cls, *exts):
        def decorator(fn):
            for ext in exts:
                cls._parsers[ext.lower()] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, ext):
        ext = ext.lower()
        if ext not in cls._parsers:
            raise ValueError(f"No parser registered for extension: {ext}")
        return cls._parsers[ext]