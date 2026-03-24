from pathlib import Path

from ml_runner.core.registries import ConfigParserRegistry
from ml_runner.core.config.utils import merge_dicts


def load_file(path):
    path = Path(path)

    parser = ConfigParserRegistry.get(path.suffix)

    return parser(path)


def load_config(cfg, seen=None):
    if seen is None:
        seen = set()

    base_path = Path(".")
    if isinstance(cfg, (str, Path)):
        path = Path(cfg).resolve()
        if path in seen:
            raise RuntimeError(f"Circular dependency detected: {path}")

        new_seen = seen.copy()
        new_seen.add(path)

        base_path = path.parent
        cfg = load_file(path)
        seen = new_seen
    elif not isinstance(cfg, dict):
        raise TypeError(f"Config must be a path or a dict, got {type(cfg)}")

    base_paths = cfg.pop("extends", [])
    if isinstance(base_paths, str):
        base_paths = [base_paths]

    final = {}

    for base in base_paths:
        base_cfg = load_config(base_path / base, seen=seen)
        final = merge_dicts(final, base_cfg)

    final = merge_dicts(final, cfg)

    return final
