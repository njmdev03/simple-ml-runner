from dataclasses import dataclass, field
from typing import Any, List, Optional, Dict
import torch
from torch.utils.data import DataLoader

from ml_runner.core.registries.service_registry import ServiceRegistry


@dataclass
class Runtime:
    """
    A rich container for the experiment's runtime components.
    Encapsulates all the objects and state needed for training and evaluation.

    Attributes:
        model: The PyTorch model instance.
        optimizer: The PyTorch optimizer instance.
        task: The task instance (BaseTask) defining training and evaluation logic.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        device: The compute device being used (e.g., 'cuda', 'cpu', 'mps').
        services: A registry for shared services provided by extensions.
        metrics: List of metric callables or objects.
        config: The full RunConfig object.
        extra: Additional metadata or state.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    task: Any
    train_loader: Optional[DataLoader] = None
    val_loader: Optional[DataLoader] = None
    device: str = "cpu"
    services: ServiceRegistry = field(default_factory=ServiceRegistry)
    metrics: List[Any] = field(default_factory=list)
    config: Any = None
    extra: Dict[str, Any] = field(default_factory=dict)
