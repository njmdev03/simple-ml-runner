from config.loader import merge_dicts

def test_deep_merge_basic():
    base = {
        "training": {"epochs": 10, "batch_size": 32},
        "optimizer": {"adam": {"lr": 0.001}}
    }
    override = {
        "training": {"epochs": 20},
        "optimizer": {"adam": {"lr": 0.01}}
    }
    merged = merge_dicts(base, override)
    assert merged["training"]["epochs"] == 20
    assert merged["training"]["batch_size"] == 32
    assert merged["optimizer"]["adam"]["lr"] == 0.01

def test_deep_merge_new_keys():
    base = {"a": {"b": 1}}
    override = {"a": {"c": 2}, "d": 3}
    merged = merge_dicts(base, override)
    assert merged == {"a": {"b": 1, "c": 2}, "d": 3}

def test_deep_merge_list_override():
    base = {"metrics": ["accuracy"]}
    override = {"metrics": ["loss", "accuracy"]}
    merged = merge_dicts(base, override)
    # Lists are not merged recursively by default in merge_dicts, they are replaced.
    assert merged["metrics"] == ["loss", "accuracy"]

def test_deep_merge_type_mismatch():
    base = {"value": {"nested": 1}}
    override = {"value": 10} # Override dict with scalar
    merged = merge_dicts(base, override)
    assert merged["value"] == 10
