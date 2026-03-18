from ml_runner.core.registries import Extension
from ml_runner.core.callbacks.timing_callback import TimingCallback
from ml_runner.core.extensions.base_extension import BaseExtension
from dataclasses import dataclass

@dataclass
class TimingConfig:
    enabled: bool = True

@Extension("timing", config_class=TimingConfig)
class TimingExtension(BaseExtension):
    def create_callbacks(self, global_config, config):
        if config and config.enabled:
            return [TimingCallback()]
        return []
