from pathlib import Path
import pytest
from unittest.mock import patch
from typing import Dict
from copy import deepcopy

from simple_config.loader import ConfigLoader, ConfigNotFoundException, UnsupportedConfigException, CircularDependencyException
from simple_config.parser.base import BaseParser
from simple_config.parser.registry import ParserRegistry


class MockParser(BaseParser):
    _configs: Dict[Path, dict] = {}

    def load(cls, path: Path):
         return cls._configs.get(path)

    def add_config(cls, path:Path, config: dict):
        cls._configs[path] = config

    def clear(cls):
        cls._configs.clear()

class MockPath(Path):
    def __init__(self, *args, exists=True):
        self._exists = exists

        super().__init__(*args)

    def exists(self, *, follow_symlinks = True):
        return self._exists


@pytest.fixture
def parser():
    mock_parser = MockParser()

    parser_reg = ParserRegistry()

    parser_reg.register(mock_parser, "test")

    return mock_parser, parser_reg


def test_resolve_path(parser):
    mock_parser, parser_reg = parser

    basic_conf = {
        "key1": 1,
        "key2": 1
    }

    mock_parser.add_config(MockPath("basic.test"), deepcopy(basic_conf))

    confl = ConfigLoader(parser_reg)

    assert confl._resolve_to_dict(MockPath("basic.test")) == basic_conf

def test_resolve_invalid_path(parser):
    _, parser_reg = parser

    confl = ConfigLoader(parser_reg)

    with pytest.raises(ConfigNotFoundException):
        confl._resolve_to_dict(MockPath("invalid.test", exists=False))

def test_resolve_no_parser(parser):
    _, parser_reg = parser

    assert parser_reg.get("invalid") == None

    confl = ConfigLoader(parser_reg)

    with pytest.raises(UnsupportedConfigException):
        confl._resolve_to_dict(MockPath("basic.invalid"))

def test_load_dict():
    basic_conf = {
        "key1": 1,
        "key2": 1
    }

    confl = ConfigLoader(ParserRegistry())

    assert confl.load_config(deepcopy(basic_conf)) == basic_conf

def test_load_dict_with_dependencies(parser):
    mock_parser, parser_reg = parser

    basic_conf = {
        "key1": 1,
        "key2": 1
    }

    basic_conf_overlay = {
        "config": [MockPath("basic.test")],
        "key2": 2
    }

    mock_parser.add_config(MockPath("basic.test"), deepcopy(basic_conf))
    mock_parser.add_config(MockPath("basic_over.test"), basic_conf_overlay)

    merged = {
        "key1": 1,
        "key2": 2,
    }

    confl = ConfigLoader(parser_reg)

    assert confl.load_config(basic_conf_overlay) == merged

def test_load_multi_dependency_dict(parser):
    mock_parser, parser_reg = parser

    dep_1 = {"key1": 2, "key2": 1}
    dep_2 = {"key1": 3, "key3": 1}
    multi_dep = {
        "config": [MockPath("dep_1.test"), MockPath("dep_2.test")],
        "key4": 1
    }

    mock_parser.add_config(MockPath("dep_1.test"), dep_1)
    mock_parser.add_config(MockPath("dep_2.test"), dep_2)

    confl = ConfigLoader(parser_reg)
    result = confl.load_config(multi_dep)

    # Order matters: dep_2 overrides dep_1, then multi_dep overrides all
    assert result["key1"] == 3
    assert result["key2"] == 1
    assert result["key3"] == 1
    assert result["key4"] == 1

def test_dependency_list_order(parser):
    mock_parser, parser_reg = parser

    dep_1 = {"val": 1}
    dep_2 = {"val": 2}

    mock_parser.add_config(MockPath("dep_1.test"), dep_1)
    mock_parser.add_config(MockPath("dep_2.test"), dep_2)

    conf = {"config": [MockPath("dep_1.test"), MockPath("dep_2.test")]}

    confl = ConfigLoader(parser_reg)
    assert confl.load_config(conf)["val"] == 2

def test_load_doesnt_mutate_configs(parser):
    mock_parser, parser_reg = parser

    dep = {"val": 1}
    mock_parser.add_config(MockPath("dep.test"), dep)

    conf = {"config": [MockPath("dep.test")], "val": 2}
    confl = ConfigLoader(parser_reg)
    confl.load_config(conf)

    assert dep == {"val": 1}
    assert conf == {"config": [MockPath("dep.test")], "val": 2}

def test_load_path(parser):
    mock_parser, parser_reg = parser
    conf = {"key": "val"}
    path = MockPath("test.test")
    mock_parser.add_config(path, conf)

    confl = ConfigLoader(parser_reg)
    assert confl.load_config(path) == conf

def test_load_string(parser):
    # Assuming load_config handles string paths by conversion to Path
    mock_parser, parser_reg = parser
    conf = {"key": "val"}
    mock_parser.add_config(MockPath("test.test"), conf)

    with patch("simple_config.loader.Path", side_effect=MockPath):
        confl = ConfigLoader(parser_reg)
        assert confl.load_config("test.test") == conf

def test_loaded_doesnt_contain_inheritance_key(parser):
    mock_parser, parser_reg = parser
    mock_parser.add_config(MockPath("dep.test"), {"a": 1})

    conf = {"config": [MockPath("dep.test")], "b": 2}
    confl = ConfigLoader(parser_reg)
    result = confl.load_config(conf)

    assert "config" not in result

def test_alternative_inheritance_key(parser):
    mock_parser, parser_reg = parser
    mock_parser.add_config(MockPath("dep.test"), {"a": 1})

    conf = {"extends": [MockPath("dep.test")], "b": 2}
    confl = ConfigLoader(parser_reg)
    result = confl.load_config(conf, inheritance_key="extends")

    assert result["a"] == 1
    assert "extends" not in result

def test_self_dependency(parser):
    mock_parser, parser_reg = parser
    path = MockPath("self.test")
    conf = {"config": [path]}
    mock_parser.add_config(path, conf)

    confl = ConfigLoader(parser_reg)
    with pytest.raises(CircularDependencyException):
        confl.load_config(path)

def test_circular_dependency_depth_1(parser):
    mock_parser, parser_reg = parser
    p1 = MockPath("1.test")
    p2 = MockPath("2.test")

    mock_parser.add_config(p1, {"config": [p2]})
    mock_parser.add_config(p2, {"config": [p1]})

    confl = ConfigLoader(parser_reg)
    with pytest.raises(CircularDependencyException):
        confl.load_config(p1)

def test_circular_dependency_depth_2(parser):
    mock_parser, parser_reg = parser
    p1 = MockPath("1.test")
    p2 = MockPath("2.test")
    p3 = MockPath("3.test")

    mock_parser.add_config(p1, {"config": [p2]})
    mock_parser.add_config(p2, {"config": [p3]})
    mock_parser.add_config(p3, {"config": [p1]})

    confl = ConfigLoader(parser_reg)
    with pytest.raises(CircularDependencyException):
        confl.load_config(p1)
