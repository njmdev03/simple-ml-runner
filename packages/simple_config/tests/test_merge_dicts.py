from simple_config.utils import merge_dicts


def test_add_item():
    dict1 = {
        "key1": 1
    }

    dict2 = {
        "key2": 2
    }

    merge_dicts(dict1, dict2)

    res = {
        "key1": 1,
        "key2": 2
    }

    assert dict1 == res

def test_base_modified_in_place():
    dict1 = {
        "key1": 1
    }

    dict2 = {
        "key2": 2
    }

    merge_dicts(dict1, dict2)

    res = {
        "key1": 1,
        "key2": 2
    }

    assert dict1 == res

def test_basic_override():
    dict1 = {
        "key1": 1,
        "key2": 2
    }

    dict2 = {
        "key2": 3
    }

    merge_dicts(dict1, dict2)

    res = {
        "key1": 1,
        "key2": 3
    }

    assert dict1 == res

def test_recursive_merge():
    dict1 = {
        "key1": 1,
        "key2": {
            "key1": 1
        },
        "key3": 1
    }

    dict2 = {
        "key1": 2,
        "key2": {
            "key1": 2
        }
    }

    merge_dicts(dict1, dict2)

    res = {
        "key1": 2,
        "key2": {
            "key1": 2
        },
        "key3": 1
    }

    assert dict1 == res

def test_values_override_dicts():
    dict1 = {
        "key1": 1,
        "key2": {
            "key1": 1
        }
    }

    dict2 = {
        "key1": 2,
        "key2": 1
    }

    merge_dicts(dict1, dict2)

    res = {
        "key1": 2,
        "key2": 1
    }

    assert dict1 == res

def test_lists_overridden():
    dict1 = {
        "list": [1, 2]
    }

    dict2 = {
        "list": [3]
    }

    merge_dicts(dict1, dict2)

    res = {
        "list": [3]
    }

    assert dict1 == res

def test_merge_lists():
    dict1 = {
        "list": [1, 2]
    }

    dict2 = {
        "list": [3]
    }

    merge_dicts(dict1, dict2, merge_lists=True)

    res = {
        "list": [1,2,3]
    }

    assert dict1 == res

def test_deep_recursive_merge():
    dict1 = {
        "key1": 1,
        "dict1": {
            "key1": 1,
            "key2": 2,
        },
        "dict2": {
            "key1": 1,
            "dict1": {
                "dict1": {
                    "key1": 1
                }
            }
        }
    }

    dict2 = {
        "dict1": {
            "key1": 10,
        },
        "dict2": {
            "key1": 10,
            "dict1": {
                "dict1": {
                    "key1": 10,
                    "key2": 2
                }
            },
            "dict2": {
                "key1": 1
            }
        }
    }

    merge_dicts(dict1, dict2, merge_lists=True)

    res = {
        "key1": 1,
        "dict1": {
            "key1": 10,
            "key2": 2,
        },
        "dict2": {
            "key1": 10,
            "dict1": {
                "dict1": {
                    "key1": 10,
                    "key2": 2
                }
            },
            "dict2": {
                "key1": 1
            }
        },
    }

    assert dict1 == res