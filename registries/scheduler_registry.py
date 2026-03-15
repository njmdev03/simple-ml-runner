# registries/schedulers.py
import torch.optim.lr_scheduler as sched

class SchedulerRegistry:
    _schedulers = {}

    @classmethod
    def register(cls, name: str):
        def decorator(fn_or_cls):
            cls._schedulers[name] = fn_or_cls
            return fn_or_cls
        return decorator

    @classmethod
    def get(cls, name: str):
        if name not in cls._schedulers:
            raise ValueError(f"No scheduler registered under name: {name}")
        return cls._schedulers[name]

    @classmethod
    def all(cls):
        return cls._schedulers.keys()


# Example built-ins
SchedulerRegistry.register("step_lr")(sched.StepLR)
SchedulerRegistry.register("cosine_annealing")(sched.CosineAnnealingLR)