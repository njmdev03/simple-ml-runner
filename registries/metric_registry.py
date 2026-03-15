# registries/metrics.py
class MetricRegistry:
    _metrics = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn):
            cls._metrics[name] = fn
            return fn
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._metrics:
            raise ValueError(f"No metric registered under name: {name}")
        return cls._metrics[name]

    @classmethod
    def all(cls):
        return cls._metrics.keys()


# Example
@MetricRegistry.register("accuracy")
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()