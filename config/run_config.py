from __future__ import annotations
import torch
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


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"


@dataclass
class DatasetConfig:
    name: str = "MNIST"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    name: str = "MLP"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str = "cross_entropy"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    epochs: int = 10
    batch_size: int = 64
    shuffle: bool = True


@dataclass
class EvaluationConfig:
    eval_during_training: bool = True
    eval_checkpoints: bool = True
    eval_frequency: int = 1
    test_on_training_data: bool = False


@dataclass
class CheckpointConfig:
    directory: Path = field(default_factory=lambda: Path("checkpoints/"))
    name: str = "model_epoch_{epoch}"
    frequency: int = 1
    save_metadata: bool = True


@dataclass
class ProfileConfig:
    enabled: bool = False
    output_path: Optional[Path] = None


@dataclass
class VisualizationConfig:
    type: List[str] = field(default_factory=lambda: ["all"])
    metrics: List[str] = field(default_factory=lambda: ["all"])
    datasets: List[str] = field(default_factory=lambda: ["testing"])
    num_samples: int = 10
    show: bool = False
    output_dir: Optional[Path] = field(default_factory=lambda: Path("vis"))
    format: str = "png"
    layout: str = "individual"


@dataclass
class RunConfig:
    """
    Parsed configuration from a file or dict.
    Matches the hierarchical structure.
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    device: str = "auto"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: Dict[str, Any] = field(default_factory=dict) # Keep raw for DataLoader init
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RunConfig":
        """
        Builds a RunConfig from a nested dictionary.
        """
        # Experiment
        exp_raw = cfg.get("experiment", {})
        if isinstance(exp_raw, str): exp_raw = {"name": exp_raw}
        experiment = ExperimentConfig(**exp_raw)

        # Dataset
        ds_raw = cfg.get("dataset", {})
        ds_name = next(iter(ds_raw)) if ds_raw else "MNIST"
        ds_params = ds_raw.get(ds_name, {})
        dataset = DatasetConfig(name=ds_name, params=ds_params)

        # Model
        m_raw = cfg.get("model", {})
        m_name = next(iter(m_raw)) if m_raw else "MLP"
        m_params = m_raw.get(m_name, {})
        model = ModelConfig(name=m_name, params=m_params)

        # Loss
        l_raw = cfg.get("loss", {})
        l_name = next(iter(l_raw)) if l_raw else "cross_entropy"
        l_params = l_raw.get(l_name, {})
        loss = LossConfig(name=l_name, params=l_params)

        # Optimizer
        opt_raw = cfg.get("optimizer", {})
        opt_name = next(iter(opt_raw)) if opt_raw else "adam"
        opt_params = opt_raw.get(opt_name, {})
        lr = opt_params.pop("lr", 0.001)
        optimizer = OptimizerConfig(name=opt_name, lr=lr, params=opt_params)

        # Profile (can be bool in YAML)
        prof_raw = cfg.get("profile", False)
        if isinstance(prof_raw, bool):
            profile = ProfileConfig(enabled=prof_raw)
        else:
            profile = ProfileConfig(**prof_raw)

        # Build the rest using standard dataclass fields if they match names
        # For simplicity in this first pass, we'll manually map the main ones
        return cls(
            experiment=experiment,
            device=cfg.get("device", "auto"),
            dataset=dataset,
            dataloader=cfg.get("dataloader", {}),
            model=model,
            loss=loss,
            optimizer=optimizer,
            metrics=cfg.get("metrics", ["accuracy"]),
            training=TrainingConfig(**cfg.get("training", {})),
            evaluation=EvaluationConfig(**cfg.get("evaluation", {})),
            checkpoint=CheckpointConfig(**cfg.get("checkpoint", {})),
            profile=profile,
            visualization=VisualizationConfig(**cfg.get("visualization", {}))
        )


@dataclass
class RuntimeConfig:
    """
    Fully-resolved runtime configuration.
    Contains objects ready to be used.
    """
    model: nn.Module
    loss_fn: nn.Module
    optimizer: Optimizer
    train_loader: Any
    val_loader: Any
    device: torch.device
    metrics: List[Callable]
    callbacks: List[Any]
    epochs: int
    profile: bool
    # Add other runtime-specific needs here