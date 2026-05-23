import pytest
from dataclasses import dataclass, field

from simple_config.builder import ConfigBuilder, MissingFieldException, VariantFormatException, ConfigTypeError
from simple_config.schema import Variant


def test_basic_schema():
    @dataclass
    class BasicConfigSchema:
        key1: int = 1
        key2: float = 0.0

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {
        "key1": 3,
        "key2": 2.5,
    }

    config = conf_builder.build(data)

    assert config.key1 == 3
    assert config.key2 == 2.5
    assert isinstance(config.key1, int)
    assert isinstance(config.key2, float)

def test_no_data():
    @dataclass
    class BasicConfigSchema:
        key1: int = 1
        key2: float = 0.0

    conf_builder = ConfigBuilder(BasicConfigSchema)

    config = conf_builder.build({})

    assert config.key1 == 1
    assert config.key2 == 0.0
    assert isinstance(config.key1, int)
    assert isinstance(config.key2, float)

def test_schema_no_default():
    @dataclass
    class BasicConfigSchema:
        key1: int
        key2: float

    conf_builder = ConfigBuilder(BasicConfigSchema)

    data = {
        "key1": 3,
        "key2": 2.5,
    }

    config = conf_builder.build(data)

    assert config.key1 == 3
    assert config.key2 == 2.5
    assert isinstance(config.key1, int)
    assert isinstance(config.key2, float)

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

    # Data is ignored if field has no type annotation (not a dataclass field)
    config = conf_builder.build(data)

    assert config.key1 == 0
    assert config.key2 == 1

def test_schema_nested_defaults():
    @dataclass
    class SubSchema:
        val: int = 10

    @dataclass
    class TopSchema:
        sub: SubSchema = field(default_factory=SubSchema)

    builder = ConfigBuilder(TopSchema)

    # Partial override of nested dataclass
    config = builder.build({"sub": {"val": 20}})
    assert config.sub.val == 20

    # Use nested default
    config = builder.build({})
    assert config.sub.val == 10

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

    assert config.key1 == 7
    assert config.key2 == 8
    assert isinstance(config.sub, SubSchema)
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

def test_variant_missing_selection():
    @dataclass
    class A:
        val: int

    @dataclass
    class Schema:
        v: Variant = Variant({"a": A})

    builder = ConfigBuilder(Schema)
    with pytest.raises(VariantFormatException):
        builder.build({"v": {}})

def test_variant_missing_selection_with_default():
    @dataclass
    class A:
        val: int = 0

    @dataclass
    class Schema:
        v: Variant = Variant({"a": A}, default="a")

    builder = ConfigBuilder(Schema)
    config = builder.build({"v": {}})

    assert isinstance(config.v, A)
    assert config.v.val == 0

def test_variant_multiple_selection():
    @dataclass
    class A:
        val: int
    @dataclass
    class B:
        val: int

    @dataclass
    class Schema:
        v: Variant = Variant({"a": A, "b": B})

    builder = ConfigBuilder(Schema)
    with pytest.raises(VariantFormatException):
        builder.build({"v": {"a": {"val": 1}, "b": {"val": 2}}})

def test_variant_no_payload():
    @dataclass
    class Simple:
        val: int = 10

    @dataclass
    class Schema:
        v: Variant = Variant({"Simple": Simple})

    builder = ConfigBuilder(Schema)

    # Key with empty dict
    config = builder.build({"v": {"Simple": {}}})
    assert isinstance(config.v, Simple)
    assert config.v.val == 10

    # Just the variant name as a string
    config = builder.build({"v": "Simple"})
    assert isinstance(config.v, Simple)
    assert config.v.val == 10

def test_invalid_type_coercion():
    @dataclass
    class Schema:
        val: int

    builder = ConfigBuilder(Schema)
    with pytest.raises(ConfigTypeError):
        builder.build({"val": "not-an-int"})
