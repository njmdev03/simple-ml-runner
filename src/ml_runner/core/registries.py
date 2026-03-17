from typing import Callable, Dict, Type



# =========================
# Built-in Registrations (optional to move later)
# =========================

from torch import nn, optim
from torch.optim import lr_scheduler


Optimizer("adam")(optim.Adam)
Optimizer("sgd")(optim.SGD)
Optimizer("adamw")(optim.AdamW)

Loss("cross_entropy")(nn.CrossEntropyLoss)
Loss("mse")(nn.MSELoss)
Loss("nll")(nn.NLLLoss)


@Metric("accuracy")
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()


Scheduler("step_lr")(lr_scheduler.StepLR)
Scheduler("cosine_annealing")(lr_scheduler.CosineAnnealingLR)


# =========================
# Generic Resolver
# =========================

def resolve_component(cfg: dict, registry: Type[BaseRegistry]):
    """
    Generic factory for registry-based components.

    Example:
        {"mlp": {"input_dim": 784}}
    """
    if not cfg:
        raise ValueError("Empty config passed to resolve_component")

    key = next(iter(cfg))
    params = cfg[key] or {}

    cls = registry.get(key)

    return cls(**params)
