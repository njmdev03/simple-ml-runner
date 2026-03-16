from registries import Extension, ExtensionRegistry
from callbacks.timing_callback import TimingCallback
from dataclasses import dataclass

@dataclass
class TimingConfig:
    enabled: bool = True

class TimingExtension(Extension):
    def get_config_class(self):
        return TimingConfig

    def create_callbacks(self, run_config, context):
        if run_config.extensions.get("timing") and run_config.extensions["timing"].enabled:
            return [TimingCallback()]
        return []

ExtensionRegistry.register("timing")(TimingExtension)
