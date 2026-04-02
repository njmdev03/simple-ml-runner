import importlib

from ml_runner.core.extensions.extension_manager import ExtensionManager
from ml_runner.core.registries.extension import ExtensionRegistry, ExtensionConfigRegistry


def _backup_and_clear_registries():
    orig_ext = ExtensionRegistry._registry.copy()
    orig_cfg = ExtensionConfigRegistry._registry.copy()
    ExtensionRegistry._registry = {}
    ExtensionConfigRegistry._registry = {}
    return orig_ext, orig_cfg


def _restore_registries(orig_ext, orig_cfg):
    ExtensionRegistry._registry = orig_ext
    ExtensionConfigRegistry._registry = orig_cfg


def test_builtin_extensions_load():
    orig_ext, orig_cfg = _backup_and_clear_registries()
    try:
        # Reload the builtin extensions package so decorators run and populate the registries
        import ml_runner.extensions as builtins
        importlib.reload(builtins)

        manager = ExtensionManager()
        # This should not raise and should construct any registered builtin extensions
        manager.register_extensions()

        names = ExtensionRegistry.all()
        assert len(names) > 0, "Expected at least one builtin extension to be registered"
        assert len(manager.get_extensions()) > 0

    finally:
        _restore_registries(orig_ext, orig_cfg)


def test_project_extensions_folder_discovery(tmp_path, monkeypatch):
    orig_ext, orig_cfg = _backup_and_clear_registries()
    try:
        project_dir = tmp_path / "project"
        ext_dir = project_dir / "extensions"
        ext_dir.mkdir(parents=True)

        plugin_file = ext_dir / "plugin_a.py"
        plugin_file.write_text(
            "from ml_runner.core.registries.extension import Extension\n"
            "from ml_runner.core.extensions.base_extension import BaseExtension\n\n"
            "@Extension('project_plugin')\n"
            "class ProjectPlugin(BaseExtension):\n"
            "    def init(self):\n"
            "        pass\n"
        )

        # Change cwd so ExtensionManager scans tmp project's ./extensions/
        monkeypatch.chdir(project_dir)

        manager = ExtensionManager()
        manager.register_extensions()

        names = [n.lower() for n in ExtensionRegistry.all()]
        assert "project_plugin" in names

    finally:
        _restore_registries(orig_ext, orig_cfg)