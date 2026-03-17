import pytest
from ml_runner.core.registries import ConfigRegistry

def test_config_registry():
    # Helper to clear parsers for testing
    ConfigRegistry._parsers = {}

    @ConfigRegistry.register(".test")
    def dummy_parser(path):
        return {"loaded": True}

    assert ConfigRegistry.get(".test") == dummy_parser
    assert ConfigRegistry.get(".TEST") == dummy_parser

    with pytest.raises(ValueError, match="No component registered"):
        ConfigRegistry.get(".unknown")

def test_config_multiple_extensions():
    ConfigRegistry._parsers = {}

    @ConfigRegistry.register(".yml", ".yaml")
    def yaml_parser(path):
        return {}

    assert ConfigRegistry.get(".yml") == yaml_parser
    assert ConfigRegistry.get(".yaml") == yaml_parser
