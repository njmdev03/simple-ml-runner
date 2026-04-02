import pytest
import yaml

@pytest.fixture
def write_config(tmp_path):
    """Fixture that returns a function to write a config file to a temporary path.

    The function can take either a dictionary (which will be dumped as YAML)
    or a string.
    """
    def _write(data, filename="config.yaml"):
        config_path = tmp_path / filename
        if isinstance(data, dict):
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f)
        else:
            # Handle strings (strip leading whitespace if any)
            import inspect
            config_path.write_text(inspect.cleandoc(data), encoding="utf-8")
        return str(config_path)

    return _write
