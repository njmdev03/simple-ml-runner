import pytest

from simple_registries.abstract_registry import AbstractClassRegistry

def test_cannot_construct():
    with pytest.raises(TypeError):
        AbstractClassRegistry()
