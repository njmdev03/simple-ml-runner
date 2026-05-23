from dataclasses import dataclass, field
from importlib.metadata import entry_points
from collections.abc import Callable

from simple_registries.registry import Registry


@dataclass(frozen=True)
class ExtensionInterface:
    name: str
    config: type
    factory: Callable

@dataclass(frozen=True)
class ExtensionNamespace:
    # Namespace extensions attach to
    name: str
    # Registry that ExtensionInterface's will be registered to
    registry: Registry = field(default_factory=Registry)

class ExtensionManager:
    def __init__(self, *namespaces: ExtensionNamespace, base_name: str = "ml_runner") -> None:
        self.namespaces: dict[str, ExtensionNamespace] = { namespace.name: namespace for namespace in namespaces }
        self.base_name: str = base_name

    def add_namespace(self, namespace: ExtensionNamespace):
        self.namespaces[namespace.name] = namespace

    def discover_extensions(self):
        for _, namespace in self.namespaces.items():
            extensions = entry_points(group=f"{self.base_name}.{namespace.name}")

            for extension in extensions:
                ext_inter: ExtensionInterface = extension.load()

                namespace.registry.register(ext_inter, ext_inter.name)

    def get_configs(self, namespace) -> list:
        return [extension.config for extension in self.namespaces[namespace].registry.all()]
