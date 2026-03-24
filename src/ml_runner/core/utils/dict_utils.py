# dict_utils.py
from typing import Any, Dict

def set_in_dict(d: Dict[str, Any], path: str, value: Any, sep: str = ".") -> None:
    """
    Sets a value in a nested dictionary using a path string.

    Example:
        cfg = {}
        set_in_dict(cfg, "optimizer.lr", 0.001)
        # cfg -> {'optimizer': {'lr': 0.001}}
    """
    keys = path.split(sep)
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value