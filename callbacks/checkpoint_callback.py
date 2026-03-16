import torch
import json
import yaml
from pathlib import Path
from log_utils import logger
from dataclasses import dataclass
from .base_callback import Callback
from config.path_utils import resolve_path_template, ensure_dir

try:
    import toml
except ImportError:
    toml = None

import configparser

# TODO: Clean up sidecar saving logic to avoid large save switches
@dataclass
class CheckpointCallback(Callback):

    path: str = "checkpoints/"
    name_template: str = "model_epoch_{epoch}"
    metadata_format: str = "pt"
    save_metadata: bool = True
    every_n_epochs: int = 1

    def on_epoch_end(self, engine):
        if engine.train_state.epoch % self.every_n_epochs == 0:
            self._save_checkpoint(engine, save_weights=True)

    def on_eval_end(self, engine):
        # We update metadata whenever evaluation finishes.
        # This keeps the metrics synced even if evaluation runs after training.
        self._save_checkpoint(engine, save_weights=False)

    def _save_checkpoint(self, engine, save_weights=True):
        # Prefer eval_state epoch if it's been set (e.g. synced from train or set manually)
        epoch = engine.eval_state.epoch if engine.eval_state.epoch > 0 else engine.train_state.epoch
        eval_metrics = self._get_eval_metrics(engine)

        # Prepare context for path resolution
        context = {
            "epoch": epoch,
            **eval_metrics
        }

        base_name = resolve_path_template(self.name_template, context)
        ckpt_filename = base_name if base_name.endswith(".pt") else f"{base_name}.pt"
        ckpt_path = Path(self.path) / ckpt_filename

        # Build current metadata payload
        train_loss = float(engine.train_state.total_loss / engine.train_state.batch) if engine.train_state.batch > 0 else None
        new_metadata = {
            "epoch": epoch,
            "checkpoint_path": ckpt_filename
        }
        if train_loss is not None: new_metadata["train_loss"] = train_loss
        if eval_metrics: new_metadata["eval_metrics"] = eval_metrics

        # Handle Saving/Editing
        if self.metadata_format == "pt":
            self._save_embedded(engine, ckpt_path, new_metadata, save_weights)
        else:
            if save_weights or not ckpt_path.exists():
                ensure_dir(str(ckpt_path))
                torch.save(engine.model.state_dict(), ckpt_path)
                logger.info(f"Saved weights to {ckpt_path}")

            if self.save_metadata:
                self._update_sidecar(base_name, new_metadata)

    def _get_eval_metrics(self, engine):
        s = engine.eval_state
        if s.batch == 0: return {}
        metrics = {"loss": float(s.total_loss / s.batch)}
        for k, v in s.total_metrics.items():
            metrics[k] = float(v / s.batch)
        return metrics

    def _save_embedded(self, engine, path, metadata, save_weights):
        # For embedded, we almost always have to save weights because they are in the same file.
        # If the file exists, we load it to merge metadata.
        payload = {"state_dict": engine.model.state_dict(), "metadata": {}}

        if path.exists():
            try:
                old_payload = torch.load(path)
                if isinstance(old_payload, dict):
                    if "state_dict" in old_payload and not save_weights:
                        payload["state_dict"] = old_payload["state_dict"]
                    if "metadata" in old_payload:
                        payload["metadata"] = old_payload["metadata"]
                else:
                    # File was a raw state_dict
                    if not save_weights: payload["state_dict"] = old_payload
            except Exception as e:
                logger.warning(f"Could not load existing checkpoint for merging: {e}")

        payload["metadata"].update(metadata)

        ensure_dir(str(path))
        torch.save(payload, path)
        logger.info(f"Updated embedded checkpoint at {path}")

    def _update_sidecar(self, base_name, metadata):
        ext = self.metadata_format.lower()
        meta_path = Path(self.path) / f"{base_name}.{ext}"

        # Load existing
        current_data = {}
        if meta_path.exists():
            try:
                current_data = self._load_sidecar(meta_path, ext)
            except Exception as e:
                logger.warning(f"Could not load sidecar for editing: {e}")

        # Merge
        current_data.update(metadata)
        if "eval_metrics" in metadata and "eval_metrics" in current_data:
             current_data["eval_metrics"].update(metadata["eval_metrics"])

        # Save
        ensure_dir(str(meta_path))
        self._write_sidecar(meta_path, current_data, ext)
        logger.info(f"Updated sidecar metadata at {meta_path}")

    def _load_sidecar(self, path, ext):
        if ext == "json":
            with open(path, 'r') as f: return json.load(f)
        elif ext == "yaml" or ext == "yml":
            with open(path, 'r') as f: return yaml.safe_load(f)
        elif ext == "toml" and toml:
            with open(path, 'r') as f: return toml.load(f)
        elif ext == "ini":
            config = configparser.ConfigParser()
            config.read(path)
            data = dict(config['METADATA'])
            if 'EVAL_METRICS' in config:
                data['eval_metrics'] = dict(config['EVAL_METRICS'])
            return data
        return {}

    def _write_sidecar(self, path, data, ext):
        if ext == "json":
            with open(path, 'w') as f: json.dump(data, f, indent=4)
        elif ext == "yaml" or ext == "yml":
            with open(path, 'w') as f: yaml.dump(data, f, default_flow_style=False)
        elif ext == "toml":
            if toml:
                with open(path, 'w') as f: toml.dump(data, f)
            else:
                with open(path.with_suffix(".json"), 'w') as f: json.dump(data, f, indent=4)
        elif ext == "ini":
            config = configparser.ConfigParser()
            # Split nested dicts for INI
            config['METADATA'] = {k: str(v) for k, v in data.items() if not isinstance(v, dict)}
            if "eval_metrics" in data:
                config['EVAL_METRICS'] = {k: str(v) for k, v in data["eval_metrics"].items()}
            with open(path, 'w') as f: config.write(f)
