from __future__ import annotations
from typing import Any, List, Optional, Dict, Type
from dataclasses import dataclass, field, is_dataclass, fields
from ml_runner.core.registries import Config, ConfigRegistry


@Config("experiment")
@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    # tags: List[str] = field(default_factory=list)


@Config("dataset")
@dataclass
class DatasetConfig:
    name: str = "MNIST"
    params: Dict[str, Any] = field(default_factory=dict)


@Config("model")
@dataclass
class ModelConfig:
    name: str = "MLP"
    params: Dict[str, Any] = field(default_factory=dict)


@Config("loss")
@dataclass
class LossConfig:
    name: str = "cross_entropy"
    params: Dict[str, Any] = field(default_factory=dict)


@Config("optimizer")
@dataclass
class OptimizerConfig:
    name: str = "adam"
    lr: float = 0.001
    params: Dict[str, Any] = field(default_factory=dict)


@Config("training")
@dataclass
class TrainingConfig:
    enabled: bool = True
    epochs: int = 10
    shuffle: bool = True
    batch_size: int = 64 # redundant with dataloader but convenient


@Config("evaluation")
@dataclass
class EvaluationConfig:
    eval_during_training: bool = True
    enabled: bool = True
    eval_checkpoints: bool = True
    eval_frequency: int = 1
    eval_on_train_data: bool = False


@Config("profile")
@dataclass
class ProfileConfig:
    enabled: bool = False
    output_path: Optional[str] = None


@Config("visualization")
@dataclass
class VisualizationConfig:
    type: List[str] = field(default_factory=lambda: ["all"])
    metrics: List[str] = field(default_factory=lambda: ["all"])
    datasets: List[str] = field(default_factory=lambda: ["eval"])
    num_samples: int = 10
    show: bool = False
    output_dir: str = "vis"
    format: str = "png"
    layout: str = "individual"


@Config("logging")
@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "train.log"
    output_dir: str = "results"


@dataclass
class RunConfig:
    """
    Parsed configuration from a file or dict.
    Matches the hierarchical structure.
    """
    device: str = "auto"
    extensions: Dict[str, Any] = field(default_factory=dict)

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # TODO: Migrate Dataloader to a DataLoaderConfig class
    dataloader: Dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # TODO: Migrate to EvaluationConfig
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], extension_config_classes: Dict[str, Type] = None) -> "RunConfig":
        if extension_config_classes is None:
            extension_config_classes = {}

        # Start with core fields from registry
        core_kwargs = {}
        core_subconfigs = cls._get_core_subconfigs()

        # 1. Process standard fields from the registry
        from dataclasses import fields
        valid_fields = {f.name for f in fields(cls)}
        for name, subcls in core_subconfigs.items():
            if name not in valid_fields:
                continue

            data = cfg.get(name, {})
            if is_dataclass(subcls):
                if isinstance(data, dict):
                    core_kwargs[name] = cls._build_dataclass(subcls, data)
                else:
                    # fallback to default instance
                    core_kwargs[name] = subcls()
            else:
                core_kwargs[name] = data

        # 2. Process explicit 'extensions' dict if present
        core_kwargs["extensions"] = cfg.get("extensions", {})

        # 3. Pull top-level non-core fields into extensions if they match an installed extension
        for key, value in cfg.items():
            if key in core_subconfigs or key == "extensions":
                continue

            # If it's a known extension, put it in extensions dict
            if key in extension_config_classes or (key + "s") in extension_config_classes:
                if key not in core_kwargs["extensions"]:
                    core_kwargs["extensions"][key] = value

            # Also handle explicit fields on RunConfig (like 'device', 'metrics')
            if key in cls.__annotations__ and key not in core_kwargs:
                core_kwargs[key] = value

        # 4. Build extension sub-configs from raw data
        for ext_name, ext_cfg_cls in extension_config_classes.items():
            # Get raw data (either in `extensions` dict or fallback to singular root name)
            raw = core_kwargs["extensions"].get(ext_name)
            if raw is None and ext_name.endswith('s'):
                raw = core_kwargs["extensions"].get(ext_name[:-1])

            if raw is None:
                core_kwargs["extensions"][ext_name] = ext_cfg_cls()
            elif isinstance(raw, dict):
                core_kwargs["extensions"][ext_name] = cls._build_dataclass(ext_cfg_cls, raw)
            elif isinstance(raw, bool):
                core_kwargs["extensions"][ext_name] = ext_cfg_cls(enabled=raw)
            else:
                # Already a dataclass or unsupported type, assign it directly
                core_kwargs["extensions"][ext_name] = raw

        return cls(**core_kwargs)

    @staticmethod
    def _build_dataclass(cls, data: Dict[str, Any]):
        """
        Recursive dataclass builder for nested configs.
        """
        from dataclasses import fields, is_dataclass

        # Regular recursive construction
        kwargs = {}
        for f in fields(cls):
            value = data.get(f.name)
            if is_dataclass(f.type) and isinstance(value, dict):
                kwargs[f.name] = RunConfig._build_dataclass(f.type, value)
            elif value is not None:
                kwargs[f.name] = value
        return cls(**kwargs)

    @staticmethod
    def _get_core_subconfigs() -> Dict[str, Type]:
        return ConfigRegistry._registry.copy()
