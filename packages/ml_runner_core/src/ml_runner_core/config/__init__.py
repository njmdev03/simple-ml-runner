from dataclasses import dataclass, field

from simple_config.schema import Variant

@dataclass
class ExperimentConfig:
    name: str = ""

@dataclass
class DataLoaderConfig:
    shuffle: bool = False

@dataclass
class EvaluationConfig:
    metrics: list[Variant]
    batch_size: int = 1
    eval_on_checkpoints: bool = True
    eval_frequency: int = 1
    enabled: bool = True

@dataclass
class TrainingConfig:
    batch_size: int = 1
    enabled: bool = True

@dataclass(kw_only=True)
class MLRunnerConfig:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    device: str # TODO: Migrate to Enum type once implemented in simple config
    task: str # TODO: More robust Variant type or Enum type
    dataset: Variant
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    model: Variant
    loss: Variant
    optimizer: Variant
    evaluation: EvaluationConfig
    training: TrainingConfig = field(default_factory=TrainingConfig)
    extensions: list[Variant]
    exports: list[Variant]
