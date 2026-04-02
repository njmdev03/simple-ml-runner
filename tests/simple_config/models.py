from dataclasses import dataclass, field
from typing import Dict, List, Optional
from simple_config import Config, Mapping

# --- Polymorphic Types Setup ---
class ModelConfig:
    pass

@Mapping(ModelConfig, "MLP")
@dataclass
class MLPConfig:
    hidden_size: int = 128
    dropout: float = 0.2

@Mapping(ModelConfig, "CNN")
@dataclass
class CNNConfig:
    filters: List[int] = field(default_factory=lambda: [32, 64])

@Config("test_poly")
@dataclass
class PolyMainConfig:
    model: ModelConfig
    optimizer: str = "adam"

# --- Nested Types Setup ---
@dataclass
class SubConfig:
    value: int

@Config("test_nested")
@dataclass
class NestedMainConfig:
    sub: SubConfig
    tags: List[str]
    metadata: Dict[str, str]
    optional_val: Optional[float] = None

@Config("test_main")
@dataclass
class MainConfig:
    model_name: str
    batch_size: int = 32
    learning_rate: float = 0.001
    layers: List[int] = field(default_factory=lambda: [64, 64])
