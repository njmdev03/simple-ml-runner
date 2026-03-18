from ml_runner.core.registries import Extension
from ml_runner.core.callbacks.tensorboard_callback import TensorboardCallback
from ml_runner.core.extensions.base_extension import BaseExtension
from dataclasses import dataclass
from ml_runner.core.config.path_utils import resolve_path_template


@dataclass
class TensorboardConfig:
    enabled: bool = False
    log_dir: str = "runs/{experiment_name}"


@Extension("tensorboard", config_class=TensorboardConfig)
class TensorboardExtension(BaseExtension):
    def create_callbacks(self, global_config, config):
        if config and config.enabled:
            context = {
                "experiment_name": global_config.experiment.name,
                "device": global_config.device
            }
            log_dir = resolve_path_template(config.log_dir, context)
            return [TensorboardCallback(log_dir=log_dir)]
        return []
