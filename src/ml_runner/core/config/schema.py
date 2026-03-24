from __future__ import annotations
from typing import Any, List, Optional, Dict, Type
from dataclasses import dataclass, field, is_dataclass, fields

from ml_runner.core.registries import Config, NamedConfig, ConfigRegistry, NamedConfigRegistry
from ml_runner.core.registries.exporter import ExporterConfigRegistry


@Config("experiment")
@dataclass
class ExperimentConfig:
    name: str = "default_experiment"


@NamedConfig("task")
class TaskConfig:
    name: str = None
    params: Dict[str, Any] = field(default_factory=dict)


@NamedConfig("model")
@dataclass
class ModelConfig:
    name: str = "MLP"
    params: Dict[str, Any] = field(default_factory=dict)


@NamedConfig("dataset")
@dataclass
class DatasetConfig:
    name: str = "MNIST"
    params: Dict[str, Any] = field(default_factory=dict)


@Config("dataloader")
@dataclass
class DataLoaderConfig:
    batch_size: int = 64
    shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    transforms: List[TransformConfig] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TransformConfig:
    name: str = None
    params: Dict[str, Any] = field(default_factory=dict)


@NamedConfig("loss")
@dataclass
class LossConfig:
    name: str = "cross_entropy"
    params: Dict[str, Any] = field(default_factory=dict)


@NamedConfig("optimizer")
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
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    batch_size: Optional[int] = None


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
    exports: Dict[str, Any] = field(default_factory=dict)

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any], extension_config_classes: Dict[str, Type] = None, exporter_config_classes: Dict[str, Type] = None) -> "RunConfig":
        # Normalize inputs
        if extension_config_classes is None:
            extension_config_classes = {}

        if exporter_config_classes is None:
            exporter_config_classes = {}

        # Start building core kwargs
        core_kwargs = {}
        core_subconfigs = cls._get_core_subconfigs()

        # Process core registered configs (from ConfigRegistry / NamedConfigRegistry)
        valid_fields = {f.name for f in fields(cls)}
        for name, subcls in core_subconfigs.items():
            if name not in valid_fields:
                continue

            data = cfg.get(name, {})
            if is_dataclass(subcls):
                if isinstance(data, dict):
                    core_kwargs[name] = cls._build_dataclass(subcls, data)
                else:
                    core_kwargs[name] = subcls()
            else:
                core_kwargs[name] = data

        # Explicit 'extensions' and 'exports' dicts
        core_kwargs["extensions"] = cfg.get("extensions", {})
        core_kwargs["exports"] = cfg.get("exports", {})

        # Pull top-level non-core fields into extensions if they match an installed extension
        for key, value in cfg.items():
            if key in core_subconfigs or key in ("extensions", "exports"):
                continue

            if key in cls.__annotations__ and key not in core_kwargs:
                core_kwargs[key] = value

        # Build extension sub-configs from raw data
        for ext_name, ext_cfg_cls in extension_config_classes.items():
            raw = core_kwargs["extensions"].get(ext_name)
            if raw is None:
                core_kwargs["extensions"][ext_name] = ext_cfg_cls()
            elif isinstance(raw, dict):
                core_kwargs["extensions"][ext_name] = cls._build_dataclass(ext_cfg_cls, raw)
            elif isinstance(raw, bool):
                core_kwargs["extensions"][ext_name] = ext_cfg_cls(enabled=raw)
            else:
                core_kwargs["extensions"][ext_name] = raw

        # Build exporter sub-configs from top-level `exports` key
        exports_raw = cfg.get('exports', {}) if isinstance(cfg, dict) else {}
        for exp_name, exp_cfg_cls in exporter_config_classes.items():
            raw = exports_raw.get(exp_name)
            if raw is None:
                core_kwargs['exports'][exp_name] = exp_cfg_cls()
            elif isinstance(raw, dict):
                core_kwargs['exports'][exp_name] = cls._build_dataclass(exp_cfg_cls, raw)
            elif isinstance(raw, bool):
                core_kwargs['exports'][exp_name] = exp_cfg_cls(enabled=raw) if hasattr(exp_cfg_cls, 'enabled') else exp_cfg_cls()
            else:
                core_kwargs['exports'][exp_name] = raw

        return cls(**core_kwargs)

    @staticmethod
    def _build_dataclass(cls, data: Dict[str, Any]):
        """
        Recursive dataclass builder for nested configs.
        Handles @NamedConfig flattening and 'params' collection.
        """
        if not isinstance(data, dict):
            return cls()

        # Handle @NamedConfig flattening: { Name: { params... } }
        if NamedConfigRegistry.contains(cls) and data:
            field_names = {f.name for f in fields(cls)}
            if len(data) == 1:
                key = next(iter(data.keys()))
                if key not in field_names and isinstance(data[key], dict):
                    name_val = key
                    inner_params = data[key]
                    data = {"name": name_val, **inner_params}

        # Regular recursive construction
        kwargs = {}
        processed_keys = set()
        for f in fields(cls):
            if f.name == "params":
                continue

            value = data.get(f.name)
            if value is not None:
                if is_dataclass(f.type) and isinstance(value, dict):
                    kwargs[f.name] = RunConfig._build_dataclass(f.type, value)
                else:
                    kwargs[f.name] = value
                processed_keys.add(f.name)

        # Collect leftover keys into 'params' if requested
        all_field_names = {f.name for f in fields(cls)}
        if "params" in all_field_names:
            params = data.get("params", {}).copy()
            for k, v in data.items():
                if k != "params" and k not in processed_keys and k not in all_field_names:
                    params[k] = v
            kwargs["params"] = params

        return cls(**kwargs)

    @staticmethod
    def _get_core_subconfigs() -> Dict[str, Type]:
        return {**ConfigRegistry._registry, **NamedConfigRegistry._registry}
