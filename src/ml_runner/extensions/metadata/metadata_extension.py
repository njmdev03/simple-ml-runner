import json
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Optional
from typing import List, Union

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.registries.callbacks import Callback
from ml_runner.core.engine.engine import Engine, EngineEvent
from ml_runner.core.utils.path_utils import resolve_path_template, ensure_dir


@dataclass
class MetadataConfig:
    enabled: bool = True
    output_file: str = "metadata.json"
    output_dir: str = "results/{experiment_name}"


@Extension("metadata", config_class=MetadataConfig)
class MetadataExtension(BaseExtension):
    """
    Extension that handles job metadata and persistence.
    Lifts logic directly into the extension.
    """
    def setup(self, event_manager: EventManager, global_config: RunConfig, config: Optional[MetadataConfig] = None):
        if config and config.enabled:
            context = {
                "experiment_name": global_config.experiment.name,
                "device": global_config.device
            }
            self.output_dir = resolve_path_template(config.output_dir, context)
            self.output_file = resolve_path_template(config.output_file, context)
            self.data: Dict[str, Any] = {}
            # attach(self, event_manager)

    @Callback(EngineEvent.JOB_START)
    def on_job_start(self, engine, **kwargs):
        # Inject this extension instance into engine so other callbacks can access its methods
        engine.metadata = self

    @Callback("record_metric")
    def on_record_metric(self, key: str, value: Any, namespace: Optional[str] = None, **kwargs):
        self.update(key, value, namespace)

    def set_flush_target(self, path: Path, format: str = "json", **kwargs):
        self.output_dir = str(path.parent)
        self.output_file = path.name

    def update(self, key: str, value: Any, namespace: Optional[str] = None):
        if namespace:
            if namespace not in self.data or not isinstance(self.data[namespace], dict):
                self.data[namespace] = {}
            self.data[namespace][key] = value
        else:
            self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @Callback(EngineEvent.EPOCH_END)
    def on_epoch_end(self, engine: Engine, **kwargs):
        self.flush()

    @Callback(EngineEvent.EVAL_END)
    def on_eval_end(self, engine, **kwargs):
        self.flush()

    @Callback(EngineEvent.JOB_END)
    def on_job_end(self, engine, **kwargs):
        self.flush()

    def flush(self):
        if (not self.data) or self.data == {}:
            return

        path = Path(self.output_dir) / self.output_file
        ensure_dir(str(path))
        ext = path.suffix.lower()[1:]

        if path.exists():
            try:
                existing = self._load(path, ext)
                self.data = self._merge(existing, self.data)
            except Exception:
                pass
        self._save(path, self.data, ext)

    def _merge(self, base, override):
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                base[k].update(v)
            else:
                base[k] = v
        return base

    def _load(self, path: Path, ext: str) -> Dict:
        if ext == "json":
            with open(path, 'r') as f: return json.load(f)
        if ext == "yaml" or ext == "yml":
            with open(path, 'r') as f: return yaml.safe_load(f)
        return {}

    def _save(self, path: Path, data: Dict, ext: str):
        if ext == "json":
            with open(path, 'w') as f: json.dump(data, f, indent=4)
        elif ext == "yaml" or ext == "yml":
            with open(path, 'w') as f: yaml.dump(data, f, default_flow_style=False)


def load_metadata_dir(path: Optional[Union[str, Path]]) -> List[Dict]:
    """Load metadata files from a directory (JSON/YAML).

    Returns a list of dicts. Files are searched recursively and returned in
    sorted order by filename. Each dict will include a `_source_path` key with
    the originating file path.
    """
    results: List[Dict] = []
    if not path:
        return results

    p = Path(path)
    if not p.exists():
        return results

    files = []
    files.extend(sorted(p.glob('**/*.json')))
    files.extend(sorted(p.glob('**/*.yaml')))
    files.extend(sorted(p.glob('**/*.yml')))

    for f in files:
        try:
            with open(f, 'r') as fh:
                if f.suffix.lower() == '.json':
                    data = json.load(fh)
                else:
                    data = yaml.safe_load(fh)
            if isinstance(data, dict):
                data['_source_path'] = str(f)
                results.append(data)
        except Exception:
            continue

    return results
