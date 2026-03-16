from __future__ import annotations
from typing import Any, Callable, List, Optional, Dict
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Optimizer
from enum import Enum
from pathlib import Path
from dataclasses import dataclass, field, fields

from registries import ModelRegistry
from registries import DatasetRegistry
from registries import OptimizerRegistry
from registries import LossRegistry
from registries import MetricRegistry
from registries import SchedulerRegistry
from callbacks.base_callback import Callback
from log_utils import logger


class LogLevel(Enum):
    INFO = "info",
    VERBOSE = "verbose"


@dataclass
class RunConfig:
    """
    Fully-typed runtime configuration for an experiment.
    Contains resolved objects ready to be consumed by the engine.
    """

    """
    Fully-typed configuration object for an experiment.
    Contains both raw values and resolved objects for immediate use.
    """
    # -------------------------------
    # General
    # -------------------------------
    name: str = "default_experiment"
    device: List = field(default_factory=["cuda", "cpu"])

    # -------------------------------
    # Dataset
    # -------------------------------
    dataset_name: str = "MNIST"
    dataset_params: dict = field(default_factory=dict)

    # -------------------------------
    # Model
    # -------------------------------
    model_name: str = "MLP"
    model_params: dict = field(default_factory=dict)

    # -------------------------------
    # Loss / Optimizer / Scheduler
    # -------------------------------
    loss_name: str = "CrossEntropyLoss"
    loss_params: dict = field(default_factory=dict)

    optimizer_name: str = "Adam"
    optimizer_params: dict = field(default_factory=dict)

    scheduler_name: Optional[str] = None
    scheduler_params: dict = field(default_factory=dict)

    # -------------------------------
    # Training hyperparameters
    # -------------------------------
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3

    # -------------------------------
    # Metrics
    # -------------------------------
    metrics: List[str] = field(default_factory=list)

    # -------------------------------
    # Callbacks
    # -------------------------------
    callbacks: List[Any] = field(default_factory=list)

    # -------------------------------
    # Logging / output
    # -------------------------------
    log_file: str = "train.log"
    save_checkpoints: bool = True
    checkpoint_path: str = "checkpoints/"
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, cfg_dict: Dict[str, Any]) -> "RunConfig":
        """
        Build a RunConfig from a dictionary while validating keys
        and normalizing types.
        """

        cfg_dict = cfg_dict.copy()

        # -------------------------------
        # Check for unknown keys
        # -------------------------------
        valid_keys = {f.name for f in fields(cls)}
        unknown = set(cfg_dict.keys()) - valid_keys

        if unknown:
            raise ValueError(
                f"Unknown config keys: {unknown}\n"
                f"Valid keys: {sorted(valid_keys)}"
            )

        # -------------------------------
        # Normalize types
        # -------------------------------

        # Convert log level string → Enum
        if "log_level" in cfg_dict and isinstance(cfg_dict["log_level"], str):
            try:
                cfg_dict["log_level"] = LogLevel[cfg_dict["log_level"].upper()]
            except KeyError:
                raise ValueError(
                    f"Invalid log_level '{cfg_dict['log_level']}'. "
                    f"Valid values: {[e.name for e in LogLevel]}"
                )

        # Convert paths
        if "log_file" in cfg_dict and not isinstance(cfg_dict["log_file"], Path):
            cfg_dict["log_file"] = Path(cfg_dict["log_file"])

        if "checkpoint_path" in cfg_dict and not isinstance(cfg_dict["checkpoint_path"], Path):
            cfg_dict["checkpoint_path"] = Path(cfg_dict["checkpoint_path"])

        return cls(**cfg_dict)


@dataclass
class RuntimeConfig:
    """
    Fully-typed configuration object for an experiment.
    Contains both raw values and resolved objects for immediate use.
    """
    # -------------------------------
    # General
    # -------------------------------
    name: str = "default_experiment"
    # seed: int = 42
    devices: List[str] = field(default_factory=["cuda", "cpu"])

    # -------------------------------
    # Dataset
    # -------------------------------
    train_dataset: Optional[Dataset] = None
    val_dataset: Optional[Dataset] = None

    # -------------------------------
    # Model
    # -------------------------------
    model: Optional[nn.Module] = None

    # -------------------------------
    # Loss / Optimizer / Scheduler
    # -------------------------------
    loss_fn: Optional[Callable] = None

    optimizer: Optional[Optimizer] = None

    scheduler: Optional[Any] = None  # Can be torch.optim.lr_scheduler or custom

    # -------------------------------
    # Training hyperparameters
    # -------------------------------
    batch_size: int = 64
    epochs: int = 10
    learning_rate: float = 1e-3

    # -------------------------------
    # Metrics
    # -------------------------------
    metrics: List[Callable] = field(default_factory=list)

    # -------------------------------
    # Callbacks
    # -------------------------------
    callbacks: List[Callback] = field(default_factory=list)

    # -------------------------------
    # Logging / output
    # -------------------------------
    log_file: Path = field(default_factory=Path("train.log"))
    save_checkpoints: bool = True
    checkpoint_path: Path = field(default_factory=Path("checkpoints/"))
    log_level: LogLevel = field(default_factory=LogLevel.INFO)

    @classmethod
    def from_run_config(config: RunConfig) -> RunConfig:
        """
        Converts a raw dictionary into a fully resolved RunConfig object,
        including dataset, model, loss, optimizer, and metrics.
        """
        cfg = RuntimeConfig()

        # -------------------------------
        # Resolve datasets
        # -------------------------------
        dataset_cls = DatasetRegistry.get(cfg.dataset_name)
        cfg.train_dataset = dataset_cls(**cfg.dataset_params, train=True)
        cfg.val_dataset = dataset_cls(**cfg.dataset_params, train=False)

        # -------------------------------
        # Resolve model
        # -------------------------------
        model_cls = ModelRegistry.get(cfg.model_name)
        cfg.model = model_cls(**cfg.model_params)

        # -------------------------------
        # Resolve loss
        # -------------------------------
        loss_cls = LossRegistry.get(cfg.loss_name)
        cfg.loss_fn = loss_cls(**cfg.loss_params)

        # -------------------------------
        # Resolve optimizer
        # -------------------------------
        optimizer_cls = OptimizerRegistry.get(cfg.optimizer_name)
        cfg.optimizer = optimizer_cls(cfg.model.parameters(), **cfg.optimizer_params)

        # -------------------------------
        # Resolve scheduler
        # -------------------------------
        if cfg.scheduler_name:
            scheduler_cls = SchedulerRegistry.get(cfg.scheduler_name)
            cfg.scheduler = scheduler_cls(cfg.optimizer, **cfg.scheduler_params)

        # -------------------------------
        # Resolve metrics
        # -------------------------------
        resolved_metrics = []
        for metric_name in cfg.metrics:
            metric_fn = MetricRegistry.get(metric_name)
            resolved_metrics.append(metric_fn)
        cfg.metrics = resolved_metrics

        return cfg