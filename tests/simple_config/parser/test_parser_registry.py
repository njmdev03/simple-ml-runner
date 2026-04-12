import pytest

from simple_config.parser.registry import ParserRegistry
from simple_config.parser.json import JSONParser
from simple_config.parser.ini import INIParser
from simple_config.parser.yaml import YAMLParser
from simple_config.parser.toml import TOMLParser


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
    ParserRegistry.clear()

    ParserRegistry.register_builtin_providers()

    assert ParserRegistry.all() != []

    assert ParserRegistry.get("ini") == INIParser
    assert ParserRegistry.get("cfg") == INIParser
    assert ParserRegistry.get("json") == JSONParser
    assert ParserRegistry.get("yml") == YAMLParser
    assert ParserRegistry.get("yaml") == YAMLParser
    assert ParserRegistry.get("toml") == TOMLParser

@pytest.mark.xdist_group(name="parser-registry")
def test_clear():
    ParserRegistry.register("json") == JSONParser

    assert list(ParserRegistry.all()) != []

    ParserRegistry.clear()

    assert list(ParserRegistry.all()) == []

@pytest.mark.xdist_group(name="parser-registry")
def test_parser_registration_is_normalized():
    ParserRegistry.clear()

    assert list(ParserRegistry.keys()) == []

    ParserRegistry.register(INIParser, ".INI")

    assert list(ParserRegistry.keys()) == ["ini"]
