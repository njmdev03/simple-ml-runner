from pathlib import Path
import pytest
from typing import Dict

from simple_config.loader import ConfigLoader, ConfigNotFoundException, UnsupportedConfigException, CircularDependencyException
from simple_config.parser.base import BaseParser
from simple_config.parser.registry import ParserRegistry, ConfigParser


@ConfigParser("test")
class MockParser(BaseParser):
    _configs: Dict[Path, dict] = {}

    @classmethod
    def load(cls, path: Path):
         return cls._configs.get(path)

    @classmethod
    def add_config(cls, path:Path, config: dict):
        cls._configs[path] = config

    @classmethod
    def clear(cls):
        cls._configs.clear()

class MockPath(Path):
    def __init__(self, *args, exists=True):
        self._exists = exists

        super().__init__(*args)

    def exists(self, *, follow_symlinks = True):
        return self._exists


basic_conf = {
    "key1": 1,
    "key2": 1
}

MockParser.add_config(MockPath("basic.test"), basic_conf)

basic_conf_overlay = {
    "config": [MockPath("basic.test")],
    "key2": 2
}

MockParser.add_config(MockPath("basic_over.test"), basic_conf_overlay)

dep_1 = {
    "key1": 2,
    "key2": 1
}

dep_2 = {
    "key1": 3,
    "key3": 1,
}

multi_dep = {
    "config": [MockPath("dep_1.test"), MockPath("dep_2.test")],
    "key1": 1,
    "key2": 1,
    "key3": 1
}

MockParser.add_config(MockPath("dep_1.test"), dep_1)
MockParser.add_config(MockPath("dep_2.test"), dep_2)
MockParser.add_config(MockPath("multi_dep.test"), multi_dep )


circular_1_depth_0 = {
    "config": [MockPath("circular1_d0.test")]
}

MockParser.add_config(MockPath("circular1_d0.test"), circular_1_depth_0)

circular_1_depth_1 = {
    "config": [MockPath("circular2_d1.test")]
}

circular_2_depth_1 = {
    "config": [MockPath("circular1_d1.test")]
}

MockParser.add_config(MockPath("circular1_d1.test"), circular_1_depth_1)
MockParser.add_config(MockPath("circular2_d1.test"), circular_2_depth_1)

circular_1_depth_2 = {
    "config": [MockPath("circular2_d2.test")]
}

circular_2_depth_2 = {
    "config": [MockPath("circular3_d2.test")]
}

circular_3_depth_2 = {
    "config": [MockPath("circular1_d2.test")]
}

MockParser.add_config(MockPath("circular1_d2.test"), circular_1_depth_2)
MockParser.add_config(MockPath("circular2_d2.test"), circular_2_depth_2)
MockParser.add_config(MockPath("circular3_d2.test"), circular_3_depth_2)


def test_resolve_path():
    assert ConfigLoader._resolve_to_dict(MockPath("basic.test")) == basic_conf

def test_resolve_invalid_path():
    with pytest.raises(ConfigNotFoundException):
        ConfigLoader._resolve_to_dict(MockPath("invalid.test", exists=False))

def test_resolve_no_parser():
    assert ParserRegistry.get("invalid") == None

    with pytest.raises(UnsupportedConfigException):
        ConfigLoader._resolve_to_dict(MockPath("basic.invalid"))


def test_load_dict():
    assert ConfigLoader.load_config(basic_conf) == basic_conf

def test_load_dict_with_dependencies():
    merged = {
        "key1": 1,
        "key2": 2,
    }

    assert ConfigLoader.load_config(basic_conf_overlay) == merged

def test_load_path():
    pass

def test_load_string():
    pass

def test_loaded_doesnt_contain_inheritance():
    pass

def test_change_inheritance_key():
    pass

def test_self_dependency():
    pass

def test_circular_dependency_depth_1():
    pass

def test_circular_dependency_depth_2():
    pass
