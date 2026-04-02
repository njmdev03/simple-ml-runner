from pathlib import Path
from typing import Union, List, Optional, Set
from simple_config.parser.registry import ParserRegistry
from simple_config.merger import merge_dicts
from simple_config.exceptions import SimpleConfigError, ConfigNotFoundError

def load_raw_config(path: Union[str, Path], seen: Optional[Set[Path]] = None) -> dict:
    """
    Load a configuration file and handle inheritance via the 'config' key.
    Detects circular dependencies.
    """
    if seen is None:
        seen = set()
        
    path = Path(path).resolve()
    
    if not path.exists():
        raise ConfigNotFoundError(f"Config file not found: {path}")
        
    if path in seen:
        raise SimpleConfigError(f"Circular dependency detected in config files: {' -> '.join(str(p) for p in seen)} -> {path}")
        
    new_seen = seen.copy()
    new_seen.add(path)
    
    try:
        parser = ParserRegistry.get_parser(str(path))
        data = parser.load(str(path))
    except Exception as e:
        raise SimpleConfigError(f"Failed to load config file '{path}': {str(e)}") from e
    
    if not isinstance(data, dict):
        raise SimpleConfigError(f"Config from '{path}' must be a dictionary, got {type(data).__name__}")

    # Step 3: Resolve 'config' recursively
    bases = data.pop("config", [])
    if isinstance(bases, str):
        bases = [bases]
    elif not isinstance(bases, list):
        raise SimpleConfigError(f"The 'config' key in '{path}' must be a string or a list of strings, got {type(bases).__name__}")
    
    result = {}
    for base_rel_path in bases:
        if not isinstance(base_rel_path, str):
            raise SimpleConfigError(f"Inheritance paths must be strings, got {type(base_rel_path).__name__} in '{path}'")
        base_path = (path.parent / base_rel_path).resolve()
        base_config = load_raw_config(base_path, seen=new_seen)
        result = merge_dicts(result, base_config)
    
    # Finally merge this config over the bases
    result = merge_dicts(result, data)
    
    return result
