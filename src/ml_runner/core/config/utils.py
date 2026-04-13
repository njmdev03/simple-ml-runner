from simple_config.utils import merge_dicts as _merge_dicts

def merge_dicts(base, override):
    """
    Backward-compatible wrapper for merge_dicts from simple_config.
    """
    return _merge_dicts(base, override)
