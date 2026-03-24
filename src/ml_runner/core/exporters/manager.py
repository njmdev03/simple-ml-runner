from typing import Dict, Type

from ml_runner.core.registries.exporter import ExporterRegistry, ExporterConfigRegistry
from ml_runner.core.exporters.base import BaseExporter
from ml_runner.core.config.schema import RunConfig


class ExporterManager:
    def __init__(self):
        self._exporters: Dict[str, BaseExporter] = {}

    def register_exporters(self):
        self._exporters = {}
        self._construct_exporters()

    def _construct_exporters(self):
        for name in ExporterRegistry.all():
            cls = ExporterRegistry.get(name)
            inst = cls()
            if hasattr(inst, 'init'):
                inst.init()
            self._exporters[name] = inst

    def get_config_classes(self) -> Dict[str, Type]:
        classes = {}
        for name in ExporterConfigRegistry.all():
            try:
                cfg_cls = ExporterConfigRegistry.get(name)
            except ValueError:
                cfg_cls = None
            if cfg_cls:
                classes[name] = cfg_cls
        return classes

    def register_cli_arguments(self, cli_registry):
        for _, exporter in self._exporters.items():
            exporter.register_cli_arguments(cli_registry)

    def run_exporter(self, name: str, run_cfg: RunConfig):
        if name not in self._exporters:
            raise ValueError(f"Exporter not found: {name}")
        exporter = self._exporters[name]
        return exporter.run(run_cfg.exports.get(name, None), run_cfg)

    def all(self):
        return list(self._exporters.keys())
