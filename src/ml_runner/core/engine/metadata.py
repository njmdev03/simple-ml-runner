import json
import yaml
import torch
from pathlib import Path
from typing import Any, Dict, Optional
import configparser
from ml_runner.core.config.path_utils import ensure_dir

try:
    import toml
except ImportError:
    toml = None

class MetadataManager:
    """
    Service for accumulating and persisting job metadata.
    """
    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._flush_path: Optional[Path] = None
        self._flush_format: str = "json"

    def set_flush_target(self, path: Path, format: str = "json"):
        self._flush_path = path
        self._flush_format = format

    def update(self, key: str, value: Any, namespace: Optional[str] = None):
        if namespace:
            if namespace not in self._data or not isinstance(self._data[namespace], dict):
                self._data[namespace] = {}
            self._data[namespace][key] = value
        else:
            self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def flush(self, path: Optional[Path] = None, format: Optional[str] = None):
        """
        Persist accumulated metadata to disk.
        """
        path = path or self._flush_path
        format = format or self._flush_format

        if not path:
            return

        ext = format.lower()

        # Merge with existing if necessary (especially for post-train eval)
        if path.exists():
            try:
                existing = self._load(path, ext)
                self._data = self._merge(existing, self._data)
            except Exception:
                pass

        ensure_dir(str(path))
        self._save(path, self._data, ext)

    def _merge(self, base, override):
        # Deep merge for eval_metrics or other namespaces
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                base[k].update(v)
            else:
                base[k] = v
        return base

    def _load(self, path: Path, ext: str) -> Dict:
        if ext == "pt":
            payload = torch.load(path)
            return payload.get("metadata", {}) if isinstance(payload, dict) else {}
        if ext == "json":
            with open(path, 'r') as f: return json.load(f)
        if ext == "yaml" or ext == "yml":
            with open(path, 'r') as f: return yaml.safe_load(f)
        return {}

    def _save(self, path: Path, data: Dict, ext: str):
        if ext == "pt":
            # For embedded, we need to preserve weights if file exists
            weights = {}
            if path.exists():
                try:
                    p = torch.load(path)
                    weights = p.get("state_dict", p) if isinstance(p, dict) else p
                except Exception: pass

            torch.save({"state_dict": weights, "metadata": data}, path)

        elif ext == "json":
            with open(path, 'w') as f: json.dump(data, f, indent=4)
        elif ext == "yaml" or ext == "yml":
            with open(path, 'w') as f: yaml.dump(data, f, default_flow_style=False)
        elif ext == "ini":
            config = configparser.ConfigParser()
            # Simple flat conversion for INI, nested dicts as sections
            for k, v in data.items():
                if isinstance(v, dict):
                    config[k.upper()] = {sk: str(sv) for sk, sv in v.items()}
                else:
                    if 'GENERAL' not in config: config['GENERAL'] = {}
                    config['GENERAL'][k] = str(v)
            with open(path, 'w') as f: config.write(f)
        elif ext == "toml" and toml:
            with open(path, 'w') as f: toml.dump(data, f)
