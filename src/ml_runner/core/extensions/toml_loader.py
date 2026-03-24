from pathlib import Path
import importlib
import importlib.util
import sys
import logging

try:
    import tomllib as _toml
except Exception:
    try:
        import toml as _toml
    except Exception:
        _toml = None


def load_toml_plugins(dir, filename: str = "ml_runner.toml"):
    """Scan candidate directories for 1-deep subfolders containing a TOML
    file named `filename`. The TOML must contain either `file` (path to a
    python file relative to the folder) or `module` (importable module name).

    Importing the referenced module/file is expected to run top-level code
    which registers exporters/extensions via decorators or manual registry
    calls.
    """
    logger = logging.getLogger(__name__)

    if _toml is None:
        logger.debug("TOML support not available; skipping toml plugin loading")
        return

    for d in dir:
        try:
            base = Path(d)
            if not base.exists() or not base.is_dir():
                continue

            parent = str(base.parent)
            added_parent = False
            if parent not in sys.path:
                sys.path.insert(0, parent)
                added_parent = True

            for child in sorted(base.iterdir()):
                try:
                    # Only consider first-level directories
                    if not child.is_dir():
                        continue

                    toml_path = child / filename
                    if not toml_path.exists() or not toml_path.is_file():
                        continue

                    try:
                        with toml_path.open("rb") as fh:
                            cfg = _toml.load(fh)
                    except Exception as e:
                        logger.warning("Failed to parse TOML %s: %s", toml_path, e)
                        continue

                    # Prefer `module` key for importable modules, otherwise `file`
                    module_name = cfg.get("module")
                    file_path = cfg.get("file") or cfg.get("path")

                    if module_name:
                        try:
                            importlib.import_module(module_name)
                            logger.info("Imported plugin module via TOML: %s", module_name)
                        except Exception as ie:
                            logger.warning("Failed to import module '%s' from %s: %s", module_name, toml_path, ie)
                        continue

                    if file_path:
                        # Resolve relative to the child folder
                        resolved = (child / file_path).resolve()
                        if not resolved.exists():
                            logger.warning("Plugin file referenced in %s not found: %s", toml_path, resolved)
                            continue

                        mod_name = f"ml_runner_plugin_{child.name}"
                        try:
                            spec = importlib.util.spec_from_file_location(mod_name, str(resolved))
                            if spec and spec.loader:
                                mod = importlib.util.module_from_spec(spec)
                                spec.loader.exec_module(mod)
                                logger.info("Imported plugin file via TOML: %s", resolved)
                        except Exception as ie:
                            logger.warning("Failed to import plugin file '%s' from %s: %s", resolved, toml_path, ie)
                        continue

                    logger.warning("TOML %s did not contain 'module' or 'file' keys", toml_path)

                except Exception as e:
                    logger.warning("Error while loading plugins from %s: %s", child, e)

            if added_parent:
                try:
                    sys.path.remove(parent)
                except ValueError:
                    pass

        except Exception as e:
            logger.warning("Error scanning candidate dir %s: %s", d, e)
