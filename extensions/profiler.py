from registries import Extension, ExtensionRegistry
from callbacks.profiler_callback import TorchProfilerCallback
from dataclasses import dataclass
from config.path_utils import resolve_path_template

@dataclass
class ProfilerConfig:
    enabled: bool = False
    output_dir: str = "profiles/{experiment_name}"
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1

class ProfilerExtension(Extension):
    def get_config_class(self):
        return ProfilerConfig

    def create_callbacks(self, run_config, context):
        cfg = run_config.extensions.get("profiler")
        if cfg and cfg.enabled:
            out_dir = resolve_path_template(cfg.output_dir, context)
            return [TorchProfilerCallback(
                output_dir=out_dir,
                wait=cfg.wait,
                warmup=cfg.warmup,
                active=cfg.active,
                repeat=cfg.repeat
            )]
        return []

ExtensionRegistry.register("profiler")(ProfilerExtension)
