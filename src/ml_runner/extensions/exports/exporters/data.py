from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import csv
import io

from ml_runner.core.registries.exporter import Exporter
from ml_runner.core.exporters.base import BaseExporter
from ml_runner.core.cli_interface.cli_argument import CLIArgument


@dataclass
class DataExportConfig:
    formats: Optional[List[str]] = None
    output: Optional[str] = None
    metadata_dir: Optional[str] = None


# We'll register handlers with the application's core registry (metadata_writer)
try:
    from ml_runner.core.registries.metadata_writer import MetadataWriterRegistry
except Exception:
    MetadataWriterRegistry = None


@Exporter("data", config_class=DataExportConfig)
class DataExporter(BaseExporter):
    """Aggregates sidecar metadata files into CSV/JSON/XLSX.

    Scans the run folder for JSON metadata files (including per-checkpoint sidecars)
    and writes combined outputs.
    """
    def run(self, cfg: Optional[DataExportConfig], global_config=None):
        # Use metadata loader if available (same semantics as plot exporter)
        metadata_dir = cfg.metadata_dir if cfg and cfg.metadata_dir else None
        output_dir = Path(cfg.output) if cfg and cfg.output else Path('.')
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            from ml_runner.extensions.metadata.metadata_extension import load_metadata_dir
            entries = load_metadata_dir(metadata_dir) if metadata_dir else []
        except Exception:
            # fallback: scan cwd
            root = Path('.')
            files = []
            files.extend(sorted(root.glob('**/*.json')))
            files.extend(sorted(root.glob('**/*.yaml')))
            files.extend(sorted(root.glob('**/*.yml')))
            entries = []
            for f in files:
                try:
                    with open(f, 'r', encoding='utf-8') as fh:
                        if f.suffix.lower() == '.json':
                            data = json.load(fh)
                        else:
                            try:
                                import yaml
                                data = yaml.safe_load(fh)
                            except Exception:
                                continue
                    if isinstance(data, dict):
                        data['_source_path'] = str(f)
                        entries.append(data)
                except Exception:
                    continue

        # normalize each metadata dict into a flat row
        rows: List[Dict[str, Any]] = []
        for md in entries:
            if not isinstance(md, dict):
                continue
            row: Dict[str, Any] = {}
            # attach source path if present
            if '_source_path' in md:
                row['_source_path'] = md.get('_source_path')
            # epoch if present
            if 'epoch' in md:
                row['epoch'] = md.get('epoch')

            for k, v in md.items():
                if k in ('_source_path', 'epoch', 'timing', 'eval_metrics'):
                    continue
                if isinstance(v, dict):
                    for nk, nv in v.items():
                        key = f"{k}.{nk}"
                        row[key] = json.dumps(nv) if isinstance(nv, (dict, list)) else nv
                elif isinstance(v, list):
                    row[k] = json.dumps(v)
                else:
                    row[k] = v

            # expand eval_metrics
            eval_metrics = md.get('eval_metrics') or {}
            if isinstance(eval_metrics, dict):
                for k, v in eval_metrics.items():
                    row[k] = v

            # expand timing
            timing = md.get('timing') or {}
            if isinstance(timing, dict):
                for k, v in timing.items():
                    row[k] = v

            rows.append(row)

        if not rows:
            return

        # compute columns (union of keys)
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        columns = ['_source_path'] if '_source_path' in all_keys else []
        if 'epoch' in all_keys:
            columns.append('epoch')
        for k in sorted(all_keys):
            if k in ('_source_path', 'epoch'):
                continue
            columns.append(k)

        # default handlers registration (if not overridden)
        def _write_json(rows, cols, out_path: Path):
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(rows, fh, indent=4)

        def _write_csv(rows, cols, out_path: Path):
            with open(out_path, 'w', newline='', encoding='utf-8') as fh:
                writer = csv.writer(fh)
                writer.writerow(cols)
                for r in rows:
                    writer.writerow([r.get(c, '') for c in cols])

        def _write_xlsx(rows, cols, out_path: Path):
            try:
                from openpyxl import Workbook
            except Exception:
                return
            wb = Workbook()
            ws = wb.active
            ws.append(cols)
            for r in rows:
                ws.append([r.get(c, '') for c in cols])
            wb.save(str(out_path))

        # import extension-local handlers so they register with the core registry
        try:
            # keep a soft dependency: importing handlers is optional
            import ml_runner.extensions.exports.exporters.data_handlers  # noqa: F401
        except Exception:
            pass

        # resolve requested formats (accept list or comma-separated string)
        raw = cfg.formats if cfg and cfg.formats else ['csv']
        if isinstance(raw, str):
            raw = [r.strip() for r in raw.split(',') if r.strip()]

        for fmt in raw:
            key = fmt.lstrip('.').lower()
            handler = None
            if MetadataWriterRegistry is not None:
                try:
                    handler = MetadataWriterRegistry.get(key)
                except Exception:
                    handler = None

            if not handler:
                # no handler registered for this format
                continue

            filename = f"metrics.{key if key != 'xlsx' else 'xlsx'}"
            try:
                handler(rows, columns, output_dir / filename)
            except Exception:
                continue

    def register_cli_arguments(self, cli_registry):
        name: str = 'data'

        cli_registry.register(CLIArgument(f"--{name}-format", config_path=f"exporters.data.format", help="Output format to use (csv,xlsx,json)"))
        cli_registry.register(CLIArgument(f"--{name}-output", config_path=f"exporters.data.output", help="Output folder for aggregated data"))
