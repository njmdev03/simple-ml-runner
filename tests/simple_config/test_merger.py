from simple_config.merger import merge_dicts

def test_merger_basic():
    """Verify recursive merging of basic dictionaries."""
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"c": 3}, "d": 4}
    merged = merge_dicts(base, override)
    assert merged == {"a": 1, "b": {"c": 3}, "d": 4}

def test_merger_list_replace():
    """Verify that lists are replaced, not merged."""
    base = {"layers": [64, 64]}
    override = {"layers": [128]}
    merged = merge_dicts(base, override)
    assert merged == {"layers": [128]}

def test_merger_nested_dict_override_scalar():
    """Verify that a dict can override a scalar and vice-versa."""
    base = {"a": 1}
    override = {"a": {"b": 2}}
    merged = merge_dicts(base, override)
    assert merged == {"a": {"b": 2}}

    base = {"a": {"b": 2}}
    override = {"a": 1}
    merged = merge_dicts(base, override)
    assert merged == {"a": 1}
