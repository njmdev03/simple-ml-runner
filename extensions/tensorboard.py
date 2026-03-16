from registries import Extension, ExtensionRegistry
from callbacks.tensorboard_callback import TensorboardCallback
from dataclasses import dataclass
from config.path_utils import resolve_path_template

@dataclass
class TensorboardConfig:
    enabled: bool = False
    log_dir: str = "runs/{experiment_name}"

class TensorboardExtension(Extension):
    def get_config_class(self):
        return TensorboardConfig

    def create_callbacks(self, run_config, context):
        cfg = run_config.extensions.get("tensorboard")
        if cfg and cfg.enabled:
            log_dir = resolve_path_template(cfg.log_dir, context)
            return [TensorboardCallback(log_dir=log_dir)]
        return []

ExtensionRegistry.register("tensorboard")(TensorboardExtension)
