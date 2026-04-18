import pytest

from simple_config.parser.registry import ParserRegistry
from simple_config.parser.json import JSONParser
from simple_config.parser.ini import INIParser
from simple_config.parser.yaml import YAMLParser
from simple_config.parser.toml import TOMLParser


def test_normalize_extensions():
    norm = ParserRegistry._normalize_extension

    assert norm(".yaml") == "yaml"
    assert norm("yaml") == "yaml"
    assert norm(".YAML") == "yaml"
    assert norm(".YaML") == "yaml"
    assert norm(".YamL") == "yaml"
    assert norm(".jSON") == "json"

def test_bundled_registrations():
    pr = ParserRegistry()

    pr.clear()

    pr.register_builtin_providers()

    assert pr.all() != []

    assert pr.get("ini") == INIParser
    assert pr.get("cfg") == INIParser
    assert pr.get("json") == JSONParser
    assert pr.get("yml") == YAMLParser
    assert pr.get("yaml") == YAMLParser
    assert pr.get("toml") == TOMLParser

def test_clear():
    pr = ParserRegistry()

    pr.register(JSONParser, "json")

    assert list(pr.all()) != []

    pr.clear()

    assert list(pr.all()) == []

def test_parser_registration_is_normalized():
    pr = ParserRegistry()

    assert list(pr.keys()) == []

    pr.register(INIParser, ".INI")

    assert list(pr.keys()) == ["ini"]
