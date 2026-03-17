from .base import BaseRegistry


class SchedulerRegistry(BaseRegistry):
    _registry = {}


def Scheduler(*names: str):
    return SchedulerRegistry.register(*names)
