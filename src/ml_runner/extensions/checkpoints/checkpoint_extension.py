from dataclasses import dataclass

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension, Config
from ml_runner.extensions.checkpoints.checkpoint_callback import CheckpointCallback
from ml_runner.core.config.path_utils import resolve_path_template, ensure_dir


@Config("checkpoint")
@dataclass
class CheckpointConfig:
    enabled: bool = True
    directory: str = "checkpoints/{experiment_name}"
    name_template: str = "ckpt_epoch_{epoch}.pt"
    metadata_format: str = "{key}:{value}"
    frequency: int = 1  # every N epochs

@Extension("checkpoints")
class CheckpointExtension(BaseExtension):
    def create_callbacks(self, run_config):
        cfg = getattr(run_config, "checkpoint", None)
        print(f"Config: {cfg}")
        if not cfg or not cfg.enabled or cfg.frequency <= 0:
            return []

        context = {
            "experiment_name": run_config.experiment.name,
            "device": run_config.device
        }

        path = resolve_path_template(cfg.directory, context)
        ensure_dir(path)

        return [
            CheckpointCallback(
                path=path,
                name_template=cfg.name_template,
                metadata_format=cfg.metadata_format,
                every_n_epochs=cfg.frequency
            )
        ]
