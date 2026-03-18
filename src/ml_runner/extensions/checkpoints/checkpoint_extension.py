from dataclasses import dataclass

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension
from ml_runner.extensions.checkpoints.checkpoint_callback import CheckpointCallback
from ml_runner.core.config.path_utils import resolve_path_template, ensure_dir


@Extension("checkpoints")
class CheckpointExtension(BaseExtension):

    @dataclass
    class Config:
        enabled: bool = True
        directory: str = "checkpoints/{experiment_name}"
        name_template: str = "ckpt_epoch_{epoch}.pt"
        metadata_format: str = "{key}:{value}"
        frequency: int = 1  # every N epochs

    def create_callbacks(self, run_config):
        # cfg = getattr(run_config, "checkpoint", None)
        print(f"Config: {run_config}")
        if not self.config or not self.config.enabled or self.config.frequency <= 0:
            return []

        context = {
            "experiment_name": run_config.experiment.name,
            "device": run_config.device
        }

        path = resolve_path_template(self.config.directory, context)
        ensure_dir(path)

        return [
            CheckpointCallback(
                path=path,
                name_template=self.config.name_template,
                metadata_format=self.config.metadata_format,
                every_n_epochs=self.config.frequency
            )
        ]
