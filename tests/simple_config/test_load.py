from pathlib import Path
import pytest
from typing import Dict
from copy import deepcopy

from simple_config.loader import ConfigLoader, ConfigNotFoundException, UnsupportedConfigException, CircularDependencyException
from simple_config.parser.base import BaseParser
from simple_config.parser.registry import ParserRegistry


class MockParser(BaseParser):
    def __init__(self):
        self._configs: Dict[Path, dict] = {}

    def load(self, path: Path):
         return self._configs.get(path)

    def add_config(self, path: Path, config: dict):
        self._configs[path] = config

    def clear(self):
        self._configs.clear()


@pytest.fixture
def parser():
    mock_parser = MockParser()

    parser_reg = ParserRegistry()

    parser_reg.register(mock_parser, "test")

    return mock_parser, parser_reg


def test_resolve_path(fs, parser):
    mock_parser, parser_reg = parser

    basic_conf = {
        "key1": 1,
        "key2": 1
    }

    path = Path("basic.test")
    fs.create_file(path)
    mock_parser.add_config(path, deepcopy(basic_conf))

    confl = ConfigLoader(parser_reg)

    assert confl._resolve_to_dict(path) == basic_conf

def test_resolve_invalid_path(fs, parser):
    _, parser_reg = parser

    confl = ConfigLoader(parser_reg)

    with pytest.raises(ConfigNotFoundException):
        confl._resolve_to_dict(Path("invalid.test"))

def test_resolve_no_parser(fs, parser):
    _, parser_reg = parser

    assert parser_reg.get("invalid") == None

    confl = ConfigLoader(parser_reg)
    path = Path("basic.invalid")
    fs.create_file(path)

    with pytest.raises(UnsupportedConfigException):
        confl._resolve_to_dict(path)

def test_load_dict():
    basic_conf = {
        "key1": 1,
        "key2": 1
    }

    confl = ConfigLoader(ParserRegistry())

    assert confl.load_config(deepcopy(basic_conf)) == basic_conf

def test_load_dict_with_dependencies(fs, parser):
    mock_parser, parser_reg = parser

    basic_conf = {
        "key1": 1,
        "key2": 1
    }

    path = Path("basic.test")
    fs.create_file(path)

    basic_conf_overlay = {
        "config": [path],
        "key2": 2
    }

    mock_parser.add_config(path, deepcopy(basic_conf))

    confl = ConfigLoader(parser_reg)

    assert confl.load_config(basic_conf_overlay) == { "key1": 1, "key2": 2 }

def test_load_multi_dependency_dict(fs, parser):
    mock_parser, parser_reg = parser

    dep_1 = {"key1": 2, "key2": 1}
    dep_2 = {"key1": 3, "key3": 1}

    p1, p2 = Path("dep_1.test"), Path("dep_2.test")
    fs.create_file(p1)
    fs.create_file(p2)

    multi_dep = {
        "config": [p1, p2],
        "key4": 1
    }

    mock_parser.add_config(p1, dep_1)
    mock_parser.add_config(p2, dep_2)

    confl = ConfigLoader(parser_reg)
    result = confl.load_config(multi_dep)

    # Order matters: dep_2 overrides dep_1, then multi_dep overrides all
    assert result["key1"] == 3
    assert result["key2"] == 1
    assert result["key3"] == 1
    assert result["key4"] == 1

def test_dependency_list_order(fs, parser):
    mock_parser, parser_reg = parser

    dep_1 = {"val": 1}
    dep_2 = {"val": 2}

    p1, p2 = Path("dep_1.test"), Path("dep_2.test")
    fs.create_file(p1)
    fs.create_file(p2)

    mock_parser.add_config(p1, dep_1)
    mock_parser.add_config(p2, dep_2)

    conf = {"config": [p1, p2]}

    confl = ConfigLoader(parser_reg)
    assert confl.load_config(conf)["val"] == 2

def test_load_doesnt_mutate_configs(fs, parser):
    mock_parser, parser_reg = parser

    dep = {"val": 1}
    path = Path("dep.test")
    fs.create_file(path)
    mock_parser.add_config(path, dep)

    conf = {"config": [path], "val": 2}
    confl = ConfigLoader(parser_reg)
    confl.load_config(conf)

    assert dep == {"val": 1}
    assert conf == {"config": [path], "val": 2}

def test_load_path(fs, parser):
    mock_parser, parser_reg = parser
    conf = {"key": "val"}
    path = Path("test.test")
    fs.create_file(path)
    mock_parser.add_config(path, conf)

    confl = ConfigLoader(parser_reg)
    assert confl.load_config(path) == conf

def test_load_string(fs, parser):
    mock_parser, parser_reg = parser
    conf = {"key": "val"}
    path = Path("test.test")
    fs.create_file(path)
    mock_parser.add_config(path, conf)

    confl = ConfigLoader(parser_reg)
    assert confl.load_config("test.test") == conf

def test_loaded_doesnt_contain_inheritance_key(fs, parser):
    mock_parser, parser_reg = parser
    path = Path("dep.test")
    fs.create_file(path)
    mock_parser.add_config(path, {"a": 1})

    conf = {"config": [path], "b": 2}
    confl = ConfigLoader(parser_reg)
    result = confl.load_config(conf)

    assert "config" not in result

def test_alternative_inheritance_key(fs, parser):
    mock_parser, parser_reg = parser
    path = Path("dep.test")
    fs.create_file(path)
    mock_parser.add_config(path, {"a": 1})

    conf = {"extends": [path], "b": 2}
    confl = ConfigLoader(parser_reg)
    result = confl.load_config(conf, inheritance_key="extends")

    assert result["a"] == 1
    assert "extends" not in result

def test_self_dependency(fs, parser):
    mock_parser, parser_reg = parser
    path = Path("self.test")
    fs.create_file(path)
    conf = {"config": [path]}
    mock_parser.add_config(path, conf)

    confl = ConfigLoader(parser_reg)
    with pytest.raises(CircularDependencyException):
        confl.load_config(path)

def test_circular_dependency_depth_1(fs, parser):
    mock_parser, parser_reg = parser
    p1 = Path("1.test")
    p2 = Path("2.test")
    fs.create_file(p1)
    fs.create_file(p2)

    mock_parser.add_config(p1, {"config": [p2]})
    mock_parser.add_config(p2, {"config": [p1]})

    confl = ConfigLoader(parser_reg)
    with pytest.raises(CircularDependencyException):
        confl.load_config(p1)

def test_circular_dependency_depth_2(fs, parser):
    mock_parser, parser_reg = parser
    p1 = Path("1.test")
    p2 = Path("2.test")
    p3 = Path("3.test")
    fs.create_file(p1)
    fs.create_file(p2)
    fs.create_file(p3)

    mock_parser.add_config(p1, {"config": [p2]})
    mock_parser.add_config(p2, {"config": [p3]})
    mock_parser.add_config(p3, {"config": [p1]})

    confl = ConfigLoader(parser_reg)
    with pytest.raises(CircularDependencyException):
        confl.load_config(p1)
