from inspect import isclass

from simple_registries.registry import Registry


"""
    Test Registering basic objects
"""
def test_registration():
    reg = Registry()
    reg.register(1, "one")

    assert len(reg.all()) > 0
    assert reg.get("one") == 1

def test_register_multiple_keys():
    reg = Registry()
    reg.register(1, "one", "One", "ONE")

    assert len(reg.all()) > 0
    assert reg.get("one") == 1
    assert reg.get("One") == 1
    assert reg.get("ONE") == 1

def test_register_dont_overwrite():
    reg = Registry()
    reg.register(1, "one")
    reg.register(2, "one", "One", "ONE")

    assert len(reg.all()) > 0
    assert reg.get("one") == 1
    assert reg.get("One") == 2
    assert reg.get("ONE") == 2

def test_register_overwrite():
    reg = Registry()
    reg.register(1, "one")
    reg.register(2, "one", "One", "ONE", overwrite=True)

    assert len(reg.all()) > 0
    assert reg.get("one") == 2
    assert reg.get("One") == 2
    assert reg.get("ONE") == 2

"""
    Test Registering classes, objects, and functions.
"""
class Item:
    pass

def item():
    pass

class Parent:
    @classmethod
    def class_child(cls):
        pass

    @staticmethod
    def static_child():
        pass

    def child(self):
        pass

def test_register_class():
    reg = Registry()
    reg.register(Item, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert isclass(reg.get("item"))

def test_register_class_instantiate():
    reg = Registry(instantiate_classes=True)
    reg.register(Item, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert isinstance(reg.get("item"), Item)

def test_register_instance():
    reg = Registry()
    it = Item()
    reg.register(it, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert isinstance(reg.get("item"), Item)
    assert reg.get("item") == it

def test_register_function():
    reg = Registry()
    reg.register(item, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert callable(reg.get("item"))
    assert reg.get("item") == item

def test_register_member_function():
    parent = Parent()

    reg = Registry()
    reg.register(parent.child, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert callable(reg.get("item"))
    assert reg.get("item") == parent.child

def test_register_static_function():
    reg = Registry()
    reg.register(Parent.static_child, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert callable(reg.get("item"))
    assert reg.get("item") == Parent.static_child

def test_register_class_function():
    reg = Registry()
    reg.register(Parent.class_child, "item")

    assert len(reg.all()) > 0
    assert reg.get("item") != None
    assert callable(reg.get("item"))
    assert reg.get("item") == Parent.class_child

"""
    Test clearing
"""
def test_registry_clear():
    reg = Registry()

    assert len(reg.all()) == 0

    reg.register(1, 1)
    reg.register(2, 2)
    reg.register(3, 3)

    assert len(reg.all()) == 3

    reg.clear()

    assert len(reg.all()) == 0

"""
    Test key fetching
"""
def test_registry_keys():
    reg = Registry()

    reg.register(1, "one")

    assert len(reg.keys()) == 1
    assert reg.keys().__contains__("one")

    reg.register(1, "One")

    assert len(reg.keys()) == 2
    assert reg.keys().__contains__("one")
    assert reg.keys().__contains__("One")
