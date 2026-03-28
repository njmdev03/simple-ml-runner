from ml_runner.core.registries.base import BaseRegistry


class MetricRegistry(BaseRegistry):
    _registry = {}


def Metric(name: str):
    """
    Decorator to register a metric with a specific name.
    """
    return MetricRegistry.register(name)
