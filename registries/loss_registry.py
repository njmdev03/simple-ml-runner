# registries/losses.py
import torch.nn as nn

class LossRegistry:
    _losses = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_or_cls):
            cls._losses[name] = fn_or_cls
            return fn_or_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._losses:
            raise ValueError(f"No loss function registered under name: {name}")
        return cls._losses[name]

    @classmethod
    def all(cls):
        return cls._losses.keys()


# Register common PyTorch losses
LossRegistry.register("cross_entropy")(nn.CrossEntropyLoss)
LossRegistry.register("mse")(nn.MSELoss)
LossRegistry.register("nll")(nn.NLLLoss)