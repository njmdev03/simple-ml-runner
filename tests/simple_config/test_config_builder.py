import pytest
from dataclasses import dataclass, field

from simple_config.builder import ConfigBuilder, MissingFieldException
from simple_config.schema import Variant


def test_basic_schema():
    @dataclass
    class BasicConfigSchema:
        key1: int = 1
        key2: float = 0

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {
        "key1": 3,
        "key2": 2,
    }

    config = conf_builder.build(data)

    assert config.key1 is not None
    assert config.key2 is not None
    assert config.key1 == 3
    assert config.key2 == 2
    assert isinstance(config.key1, int)
    # assert isinstance(config.key2, float)

def test_no_data():
    @dataclass
    class BasicConfigSchema:
        key1: int = 1
        key2: float = 0

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {}

    config = conf_builder.build(data)

    assert config.key1 is not None
    assert config.key2 is not None
    assert config.key1 == 1
    assert config.key2 == 0
    assert isinstance(config.key1, int)
    # assert isinstance(config.key2, float)

def test_schema_no_default():
    @dataclass
    class BasicConfigSchema:
        key1: int
        key2: float

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {
        "key1": 3,
        "key2": 2,
    }

    config = conf_builder.build(data)

    assert config.key1 is not None
    assert config.key2 is not None
    assert config.key1 == 3
    assert config.key2 == 2
    assert isinstance(config.key1, int)
    # assert isinstance(config.key2, float)

def test_schema_no_default_no_data():
    @dataclass
    class BasicConfigSchema:
        key1: int
        key2: float

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {}

    with pytest.raises(MissingFieldException):
        conf_builder.build(data)

def test_schema_no_type_no_data():
    @dataclass
    class BasicConfigSchema:
        key1 = 0
        key2 = 1

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {}

    config = conf_builder.build(data)

    assert config.key1 is not None
    assert config.key2 is not None
    assert config.key1 == 0
    assert config.key2 == 1

def test_schema_no_type():
    @dataclass
    class BasicConfigSchema:
        key1 = 0
        key2 = 1

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {
        "key1": 1,
        "key2": 3
    }

    config = conf_builder.build(data)

    assert config.key1 == 0
    assert config.key2 == 1

def test_schema_contains_dataclass():
    @dataclass
    class SubSchema:
        key1: int = 3
        key2: int = 4

    @dataclass
    class TopConfigSchema:
        sub: SubSchema
        key1: int = 0
        key2: int = 1

    conf_builder = ConfigBuilder(TopConfigSchema)

    data = {
        "sub": {
            "key1": 5,
            "key2": 6
        },
        "key1": 7,
        "key2": 8
    }

    config = conf_builder.build(data)

    assert config.key1 is not None
    assert config.key2 is not None
    assert config.key1 == 7
    assert config.key2 == 8
    assert config.sub.key1 == 5
    assert config.sub.key2 == 6

def test_variant_fields():
    @dataclass
    class Rectangle:
        width: int
        height: int

    @dataclass
    class Circle:
        radius: int

    @dataclass
    class ConfigSchema:
        shape: Variant = Variant({
            "Rectangle": Rectangle,
            "Circle": Circle
        })
        key1: int = 0
        key2: int = 1

    conf_builder = ConfigBuilder(ConfigSchema)

    data = {
        "shape": {
            "Rectangle" : {
                "width": 6,
                "height": 9
            }
        },
        "key1": 1,
        "key2": 3
    }

    config = conf_builder.build(data)

    assert config.key1 == 1
    assert config.key2 == 3
    assert isinstance(config.shape, Rectangle)
    assert config.shape.width == 6
    assert config.shape.height == 9
