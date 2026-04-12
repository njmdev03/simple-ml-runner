import pytest

from simple_config.parser.registry import ParserRegistry
from simple_config.parser.json import JSONParser


@pytest.mark.xdist_group(name="parser-registry")
def test_normalize_extensions():
    norm = ParserRegistry._normalize_extension

    assert norm(".yaml") == "yaml"
    assert norm("yaml") == "yaml"
    assert norm(".YAML") == "yaml"
    assert norm(".YaML") == "yaml"
    assert norm(".YamL") == "yaml"
    assert norm(".jSON") == "json"

@pytest.mark.xdist_group(name="parser-registry")
def test_bundled_registrations():
    print(ParserRegistry.all())

    # ParserRegistry.clear()
    # ParserRegistry.register(JSONParser, "json")
    assert ParserRegistry.get("json") == JSONParser
