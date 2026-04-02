import pytest
from simple_config import build_config
from simple_config.exceptions import ConfigValidationError, ConfigTypeError

def test_missing_required_field(write_config):
    """Verify that missing required fields raise ConfigValidationError with a descriptive message."""
    config_path = write_config({"batch_size": 128}) # model_name is missing

    with pytest.raises(ConfigValidationError, match="model_name"):
        build_config("test_main", config_path)

def test_type_error_with_path(write_config):
    """Verify that type errors include the full field path for easier debugging."""
    config_path = write_config({
        "sub": {"value": "not-an-int"},
        "tags": ["a", "b"],
        "metadata": {}
    })

    with pytest.raises(ConfigTypeError) as excinfo:
        build_config("test_nested", config_path)

    assert "At 'sub.value'" in str(excinfo.value)
    assert "Expected type 'int'" in str(excinfo.value)

def test_list_type_error(write_config):
    """Verify that list type errors include the index and field path."""
    config_path = write_config({
        "sub": {"value": 1},
        "tags": ["a", 1],
        "metadata": {}
    })

    with pytest.raises(ConfigTypeError) as excinfo:
        build_config("test_nested", config_path)

    assert "At 'tags[1]'" in str(excinfo.value)
    assert "Expected type 'str'" in str(excinfo.value)

def test_friendly_schema_error(write_config):
    """Verify that polymorphic types require exactly one implementation key."""
    # Multiple keys in a polymorphic field
    config_path = write_config({
        "model": {
            "MLP": {},
            "CNN": {}
        }
    })

    with pytest.raises(ConfigValidationError, match="exactly one key"):
        build_config("test_poly", config_path)

def test_unregistered_schema():
    """Verify that building an unregistered schema raises an appropriate error."""
    with pytest.raises(Exception, match="Config schema 'non_existent' not found"):
        build_config("non_existent", "dummy.yaml")
