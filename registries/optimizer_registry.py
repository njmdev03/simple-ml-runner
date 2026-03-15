# registries/optimizers.py
import torch.optim as optim

class OptimizerRegistry:
    _optimizers = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_or_cls):
            cls._optimizers[name] = fn_or_cls
            return fn_or_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._optimizers:
            raise ValueError(f"No optimizer registered under name: {name}")
        return cls._optimizers[name]

    @classmethod
    def all(cls):
        return cls._optimizers.keys()


# Pre-register built-in PyTorch optimizers
OptimizerRegistry.register("adam")(optim.Adam)
OptimizerRegistry.register("sgd")(optim.SGD)
OptimizerRegistry.register("adamw")(optim.AdamW)