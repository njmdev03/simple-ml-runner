from tests.simple_config.models import (
    MainConfig,
    PolyMainConfig,
    MLPConfig,
)
from simple_config import build_config

def test_build_config(write_config):
    """Verify that a basic configuration can be built into a dataclass."""
    config_path = write_config({
        "model_name": "transformer",
        "batch_size": 128,
        "layers": [256, 256, 256]
    })

    cfg = build_config("test_main", config_path)

    assert isinstance(cfg, MainConfig)
    assert cfg.model_name == "transformer"
    assert cfg.batch_size == 128
    assert cfg.layers == [256, 256, 256]

def test_polymorphic_build(write_config):
    """Verify that polymorphic mappings (Friendly Schema) work as expected."""
    config_path = write_config({
        "model": {
            "MLP": {
                "hidden_size": 256
            }
        }
    })

    cfg = build_config("test_poly", config_path)
    assert isinstance(cfg, PolyMainConfig)
    assert isinstance(cfg.model, MLPConfig)
    assert cfg.model.hidden_size == 256
    assert cfg.model.dropout == 0.2

def test_nested_complex_build(write_config):
    """Verify building of nested dataclasses and complex types like List and Dict."""
    config_path = write_config({
        "sub": {"value": 42},
        "tags": ["a", "b"],
        "metadata": {"key": "val"},
        "optional_val": 0.5
    })

    cfg = build_config("test_nested", config_path)
    assert cfg.sub.value == 42
    assert cfg.tags == ["a", "b"]
    assert cfg.metadata == {"key": "val"}
    assert cfg.optional_val == 0.5

def test_float_accepts_int(write_config):
    """Verify that float fields can accept integers and perform conversion."""
    config_path = write_config({
        "sub": {"value": 1},
        "tags": [],
        "metadata": {},
        "optional_val": 1
    })

    cfg = build_config("test_nested", config_path)
    assert cfg.optional_val == 1.0
    assert isinstance(cfg.optional_val, float)
