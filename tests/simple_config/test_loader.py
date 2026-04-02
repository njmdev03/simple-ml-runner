import pytest
import os
from pathlib import Path
from simple_config.loader import load_raw_config
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError

def test_config_inheritance(write_config):
    """Verify that configuration inheritance works using the write_config fixture."""
    # Write base and override configs
    write_config({"model_name": "resnet", "batch_size": 64}, filename="base.yaml")
    over_path = write_config({"config": "base.yaml", "learning_rate": 0.01, "batch_size": 128}, filename="over.yaml")

    # Change CWD to the temporary directory so relative paths are resolved correctly
    old_cwd = os.getcwd()
    os.chdir(Path(over_path).parent)
    try:
        # Load the overriding config
        raw = load_raw_config("over.yaml")

        # Verify merged results
        assert raw["model_name"] == "resnet"
        assert raw["batch_size"] == 128
        assert raw["learning_rate"] == 0.01
        assert "config" not in raw
    finally:
        os.chdir(old_cwd)

def test_circular_dependency(write_config):
    """Verify that circular dependencies are detected."""
    # We use filenames to create the cycle
    write_config({"config": "b.yaml"}, filename="a.yaml")
    write_config({"config": "a.yaml"}, filename="b.yaml")

    # Change CWD to resolve relative paths in 'config' key
    import os
    from pathlib import Path
    old_cwd = os.getcwd()
    os.chdir(Path(write_config({}, filename="dummy.yaml")).parent)
    try:
        with pytest.raises(SimpleConfigError, match="Circular dependency detected"):
            load_raw_config("a.yaml")
    finally:
        os.chdir(old_cwd)

def test_missing_file():
    """Verify that a missing file raises ConfigNotFoundError."""
    with pytest.raises(ConfigNotFoundError):
        load_raw_config("non_existent_file.yaml")

def test_custom_inheritance_key(write_config):
    """Verify that a custom inheritance key can be used instead of 'config'."""
    base_path = write_config({"model_name": "resnet"}, filename="base.yaml")
    over_path = write_config({"extends": "base.yaml", "batch_size": 128}, filename="over.yaml")

    # Change CWD to resolve relative paths
    import os
    from pathlib import Path
    old_cwd = os.getcwd()
    os.chdir(Path(over_path).parent)
    try:
        raw = load_raw_config("over.yaml", inheritance_key="extends")
        assert raw["model_name"] == "resnet"
        assert raw["batch_size"] == 128
        assert "extends" not in raw
    finally:
        os.chdir(old_cwd)
