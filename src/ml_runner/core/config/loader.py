from typing import Union, List, Optional, Any
from pathlib import Path
from simple_config.loader import load_raw_config as _load_raw_config
from simple_config.utils import merge_dicts as _merge_dicts

def load_config(path: Union[str, Path]) -> dict:
    """
    Backward-compatible wrapper for load_raw_config from simple_config.
    """
    return _load_raw_config(path)

def merge_dicts(base: dict, override: dict) -> dict:
    """
    Backward-compatible wrapper for merge_dicts from simple_config.
    """
    return _merge_dicts(base, override)
