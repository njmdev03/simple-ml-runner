import pytest
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml
import os

from simple_config import build_config, Config, ConfigParser
from simple_config.merger import merge_dicts
from simple_config.loader import load_raw_config
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError, ConfigValidationError

@Config("test_main")
@dataclass
class MainConfig:
    model_name: str
    batch_size: int = 32
    learning_rate: float = 0.001
    layers: List[int] = field(default_factory=lambda: [64, 64])
    extensions: Dict[str, Any] = field(default_factory=dict)

@Config("extension.wandb")
@dataclass
class WandbConfig:
    project: str
    entity: Optional[str] = None

def test_merger_basic():
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"c": 3}, "d": 4}
    merged = merge_dicts(base, override)
    assert merged == {"a": 1, "b": {"c": 3}, "d": 4}

def test_merger_list_replace():
    base = {"layers": [64, 64]}
    override = {"layers": [128]}
    merged = merge_dicts(base, override)
    assert merged == {"layers": [128]}

def test_config_inheritance(tmp_path):
    base_file = tmp_path / "base.yaml"
    base_file.write_text("model_name: resnet\nbatch_size: 64", encoding="utf-8")
    
    over_file = tmp_path / "over.yaml"
    over_file.write_text(f"config: {base_file.name}\nlearning_rate: 0.01", encoding="utf-8")
    
    # Change CWD to tmp_path to resolve relative path
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        raw = load_raw_config("over.yaml")
        assert raw["model_name"] == "resnet"
        assert raw["batch_size"] == 64
        assert raw["learning_rate"] == 0.01
        assert "config" not in raw
    finally:
        os.chdir(old_cwd)

def test_build_config(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model_name: transformer
batch_size: 128
layers: [256, 256, 256]
wandb:
  project: my-project
""", encoding="utf-8")
    
    cfg = build_config("test_main", str(config_file))
    
    assert isinstance(cfg, MainConfig)
    assert cfg.model_name == "transformer"
    assert cfg.batch_size == 128
    assert cfg.layers == [256, 256, 256]
    assert "wandb" in cfg.extensions
    assert isinstance(cfg.extensions["wandb"], WandbConfig)
    assert cfg.extensions["wandb"].project == "my-project"

def test_circular_dependency(tmp_path):
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(f"config: {b.name}", encoding="utf-8")
    b.write_text(f"config: {a.name}", encoding="utf-8")
    
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        with pytest.raises(SimpleConfigError, match="Circular dependency detected"):
            load_raw_config("a.yaml")
    finally:
        os.chdir(old_cwd)

def test_missing_required_field(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("batch_size: 128", encoding="utf-8") # model_name is missing
    
    with pytest.raises(ConfigValidationError, match="model_name"):
        build_config("test_main", str(config_file))
