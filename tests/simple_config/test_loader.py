import pytest
import shutil
import os
from pathlib import Path
from simple_config.loader import load_raw_config
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError

# Path to our static test configurations
CONFIGS_DIR = Path(__file__).parent / "configs"

@pytest.fixture
def test_configs(tmp_path):
    """Fixture to copy static configurations to a temporary directory."""
    temp_dir = tmp_path / "configs"
    temp_dir.mkdir()
    for f in CONFIGS_DIR.glob("*.yaml"):
        shutil.copy(f, temp_dir / f.name)
    return temp_dir

def test_config_inheritance_from_file(test_configs):
    """Verify that configuration inheritance works from static files."""
    # Change CWD to the temporary directory so relative paths are resolved correctly
    old_cwd = os.getcwd()
    os.chdir(test_configs)
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

def test_circular_dependency(tmp_path):
    """Verify that circular dependencies are detected."""
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

def test_missing_file():
    """Verify that a missing file raises ConfigNotFoundError."""
    with pytest.raises(ConfigNotFoundError):
        load_raw_config("non_existent_file.yaml")
