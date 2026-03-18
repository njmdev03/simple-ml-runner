from __future__ import annotations
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
from ml_runner.core.registries import ConfigRegistry


@dataclass
class ExperimentConfig:
    name: str = "default_experiment"
    # tags: List[str] = field(default_factory=list)


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
    enabled: bool = True
    epochs: int = 10
    shuffle: bool = True
    batch_size: int = 64 # redundant with dataloader but convenient


@dataclass
class EvaluationConfig:
    eval_during_training: bool = True
    enabled: bool = True
    eval_checkpoints: bool = True
    eval_frequency: int = 1
    eval_on_train_data: bool = False


@dataclass
class ProfileConfig:
    enabled: bool = False
    output_path: Optional[str] = None


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
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    device: str = "auto"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: Dict[str, Any] = field(default_factory=dict)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    # checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    extensions: Dict[str, Any] = field(default_factory=dict)

    @property
    def do_train(self) -> bool:
        return self.training.enabled

    @property
    def do_eval(self) -> bool:
        return self.evaluation.enabled

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RunConfig":
        """
        Builds a RunConfig from a nested dictionary.
        """
        from ml_runner.core.registries import ExtensionRegistry

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

        # Profile
        prof_raw = cfg.get("profile", {})
        if isinstance(prof_raw, bool):
            profile = ProfileConfig(enabled=prof_raw)
        else:
            profile = ProfileConfig(**prof_raw)

        # Eval
        eval_dict = cfg.get("eval", cfg.get("evaluation", {}))
        # Handle 'test_on_training_data' -> 'eval_on_train_data'
        if "test_on_training_data" in eval_dict:
            eval_dict["eval_on_train_data"] = eval_dict.pop("test_on_training_data")

        from dataclasses import is_dataclass, fields

        def build_dataclass(cls, data: dict):
            kwargs = {}

            for f in fields(cls):
                value = data.get(f.name)

                if is_dataclass(f.type) and isinstance(value, dict):
                    kwargs[f.name] = build_dataclass(f.type, value)
                elif value is not None:
                    kwargs[f.name] = value

            return cls(**kwargs)

        ext_instances = {}

        for name in ExtensionRegistry.all():
            ext_cls = ExtensionRegistry.get(name)

            config_cls = getattr(ext_cls, "Config", None)

            if config_cls is None:
                # No config → just construct
                ext = ext_cls(global_config=None, config=None)
            else:
                raw = cfg.get(name, {})

                if isinstance(raw, bool):
                    config = config_cls(enabled=raw)
                elif isinstance(raw, dict):
                    config = build_dataclass(config_cls, raw)
                else:
                    config = config_cls()

                ext = ext_cls(global_config=None, config=config)

            ext_instances[name] = ext

        config = cls(
            experiment=experiment,
            device=cfg.get("device", "auto"),
            dataset=dataset,
            dataloader=cfg.get("dataloader", {}),
            model=model,
            loss=loss,
            optimizer=optimizer,
            metrics=cfg.get("metrics", ["accuracy"]),
            training=TrainingConfig(**cfg.get("training", {})),
            evaluation=EvaluationConfig(**eval_dict),
            # checkpoint=CheckpointConfig(**cfg.get("checkpoint", {})),
            profile=profile,
            visualization=VisualizationConfig(**cfg.get("visualization", {})),
            logging=LoggingConfig(**cfg.get("logging", {})),
            extensions={}
        )

        for ext in ext_instances.values():
            ext.global_config = config

        config.extensions = ext_instances

        return config
