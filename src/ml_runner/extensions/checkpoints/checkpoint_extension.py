from dataclasses import dataclass

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension
from ml_runner.extensions.checkpoints.checkpoint_callback import CheckpointCallback
from ml_runner.core.config.path_utils import resolve_path_template, ensure_dir
from ml_runner.core.config.schema import RunConfig

@dataclass
class Config:
    enabled: bool = True
    directory: str = "checkpoints/{experiment_name}"
    name_template: str = "ckpt_epoch_{epoch}.pt"
    metadata_format: str = "{key}:{value}"
    frequency: int = 1  # every N epochs

@Extension("checkpoints", config_class=Config)
class CheckpointExtension(BaseExtension):

    def create_callbacks(self, config: Config, global_config: RunConfig):
        if not config or not config.enabled or config.frequency <= 0:
            return []

        context = {
            "experiment_name": global_config.experiment.name,
            "device": global_config.device
        }

        path = resolve_path_template(config.directory, context)
        ensure_dir(path)

        return [
            CheckpointCallback(
                path=path,
                name_template=config.name_template,
                metadata_format=config.metadata_format,
                every_n_epochs=config.frequency
            )
        ]
