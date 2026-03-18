from ml_runner.core.registries import Extension
from ml_runner.core.callbacks.profiler_callback import TorchProfilerCallback
from ml_runner.core.extensions.base_extension import BaseExtension
from dataclasses import dataclass
from ml_runner.core.config.path_utils import resolve_path_template


@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_dir: str = "profiles/{experiment_name}"
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1


@Extension("profiler", config_class=ProfilerConfig)
class ProfilerExtension(BaseExtension):
    def create_callbacks(self, global_config, config):
        if config and config.enabled:
            context = {
                "experiment_name": global_config.experiment.name,
                "device": global_config.device
            }
            out_dir = resolve_path_template(config.output_dir, context)
            return [TorchProfilerCallback(
                output_dir=out_dir,
                wait=config.wait,
                warmup=config.warmup,
                active=config.active,
                repeat=config.repeat
            )]
        return []
