import pytest
from simple_config import build_config
from simple_config.exceptions import ConfigValidationError, ConfigTypeError

def test_missing_required_field(tmp_path):
    """Verify that missing required fields raise ConfigValidationError with a descriptive message."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("batch_size: 128", encoding="utf-8") # model_name is missing

    with pytest.raises(ConfigValidationError, match="model_name"):
        build_config("test_main", str(config_file))

def test_type_error_with_path(tmp_path):
    """Verify that type errors include the full field path for easier debugging."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
sub:
  value: "not-an-int"
tags: ["a", "b"]
metadata: {}
""", encoding="utf-8")

    with pytest.raises(ConfigTypeError) as excinfo:
        build_config("test_nested", str(config_file))

    assert "At 'sub.value'" in str(excinfo.value)
    assert "Expected type 'int'" in str(excinfo.value)

def test_list_type_error(tmp_path):
    """Verify that list type errors include the index and field path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
sub: {value: 1}
tags: ["a", 1]
metadata: {}
""", encoding="utf-8")

    with pytest.raises(ConfigTypeError) as excinfo:
        build_config("test_nested", str(config_file))

    assert "At 'tags[1]'" in str(excinfo.value)
    assert "Expected type 'str'" in str(excinfo.value)

def test_friendly_schema_error(tmp_path):
    """Verify that polymorphic types require exactly one implementation key."""
    # Multiple keys in a polymorphic field
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
model:
  MLP: {}
  CNN: {}
""", encoding="utf-8")

    with pytest.raises(ConfigValidationError, match="exactly one key"):
        build_config("test_poly", str(config_file))

def test_unregistered_schema():
    """Verify that building an unregistered schema raises an appropriate error."""
    with pytest.raises(Exception, match="Config schema 'non_existent' not found"):
        build_config("non_existent", "dummy.yaml")
