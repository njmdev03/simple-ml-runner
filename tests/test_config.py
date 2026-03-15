import pytest

from config.loader import merge_dicts
from config.loader import load_config


def test_dict_merge_basic():
    base = {
        "model": {"hidden_dim": 128},
        "trainer": {"epochs": 5}
    }

    override = {
        "model": {"hidden_dim": 256}
    }

    merged = merge_dicts(base, override)

    assert merged["model"]["hidden_dim"] == 256
    assert merged["trainer"]["epochs"] == 5


def test_dict_merge_nested():
    base = {
        "metrics": {
            "train": {
                "loss": True,
                "accuracy": True
            },
            "eval": {
                "loss": True
            }
        }
    }

    override = {
        "metrics": {
            "eval": {
                "accuracy": True
            }
        }
    }

    merged = merge_dicts(base, override)

    assert merged["metrics"]["eval"]["loss"] is True
    assert merged["metrics"]["eval"]["accuracy"] is True
    assert merged["metrics"]["train"]["accuracy"] is True


def test_list_override():
    base = {
        "callbacks": ["console_logger", "checkpoint"]
    }

    override = {
        "callbacks": ["tensorboard"]
    }

    merged = merge_dicts(base, override)

    assert merged["callbacks"] == ["tensorboard"]


def test_config_loading():
    cfg = load_config("tests/configs/experiment.yaml")

    assert cfg["model"]["hidden_dim"] == 256
    assert cfg["dataset"]["batch_size"] == 64
    assert cfg["optimizer"]["lr"] == 0.0005
    assert cfg["trainer"]["device"] == "cuda"


def test_config_callbacks():
    cfg = load_config("tests/configs/experiment.yaml")

    assert "console_logger" in cfg["callbacks"]
    assert "checkpoint" in cfg["callbacks"]


def test_circular_extends():
    with pytest.raises(RuntimeError):
        load_config("tests/configs/bad_config.yaml")