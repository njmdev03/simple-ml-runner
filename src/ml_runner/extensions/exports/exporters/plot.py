from dataclasses import dataclass
from typing import Optional, List
from pathlib import Path

from ml_runner.core.registries.exporter import Exporter
from ml_runner.core.cli_interface.cli_argument import CLIArgument
from ml_runner.core.exporters.base import BaseExporter
from ml_runner.core.log_utils import logger


@dataclass
class ExportPlotConfig:
    plots: Optional[list] = None
    output: Optional[str] = None
    metadata_dir: Optional[str] = None


@Exporter("plot", config_class=ExportPlotConfig)
class PlotExporter(BaseExporter):
    """Simple exporter that reads persisted metrics/metadata and writes CSV/PNG plots.

    This exporter is intentionally minimal and operates only on files under a
    run folder (e.g. output/results/<run>/).
    """
    def run(self, cfg: Optional[ExportPlotConfig], global_config=None):
        # Lazy import matplotlib; require it for plotting
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception:
            logger.error("Matplotlib not found, skipping plotting")
            return

        # Determine metadata directory and output
        metadata_dir = cfg.metadata_dir if cfg and cfg.metadata_dir else None
        output_dir = Path(cfg.output) if cfg and cfg.output else Path('.')

        # Parse plot specifications. Support repeated flags and comma-separated lists.
        raw_plots = cfg.plots
        if isinstance(raw_plots, str):
            raw_plots = [raw_plots]

        plot_groups: List[List[str]] = []
        for item in raw_plots:
            if isinstance(item, str):
                parts = [p.strip() for p in item.split(',') if p.strip()]
                if parts:
                    plot_groups.append(parts)
            elif isinstance(item, (list, tuple)):
                parts = [str(p).strip() for p in item if str(p).strip()]
                if parts:
                    plot_groups.append(parts)

        # Load metadata files using the metadata extension helper if available
        entries = []
        try:
            from ml_runner.extensions.metadata.metadata_extension import load_metadata_dir
            entries = load_metadata_dir(metadata_dir) if metadata_dir else []
        except Exception:
            logger.error("Metadata extension could not be loaded. Cannot import metadata, exiting.")
            return

        # Normalize entries into rows for plotting
        rows = []
        for md in entries:
            if not isinstance(md, dict):
                continue
            epoch = md.get('epoch')
            if epoch is None:
                continue

            row = {'epoch': epoch}

            # include top-level scalar metrics (excluding special keys)
            for k, v in md.items():
                if k in ('epoch', 'eval_metrics', 'timing', '_source_path'):
                    continue
                row[k] = v

            # expand eval_metrics into top-level keys
            eval_metrics = md.get('eval_metrics') or {}
            if isinstance(eval_metrics, dict):
                for k, v in eval_metrics.items():
                    row[k] = v

            # expand timing fields (expose train/eval durations too)
            timing = md.get('timing') or {}
            if isinstance(timing, dict):
                for k, v in timing.items():
                    row[k] = v

            rows.append(row)

        # nothing to do
        if not rows:
            return

        # Ensure output dir exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # plotting helpers
        def save_fig(fig, name: str):
            out = output_dir / f"{name}.png"
            try:
                fig.savefig(str(out), bbox_inches='tight')
            finally:
                plt.close(fig)

        rows_sorted = sorted(rows, key=lambda x: x['epoch'])
        epochs = [r['epoch'] for r in rows_sorted]

        def metric_to_series(metric: str):
            m = metric.lower()
            if m == 'loss':
                return [
                    ('train_loss', 'training loss'),
                    ('loss', 'evaluation loss')
                ]
            return [(metric, metric)]

        # For each requested plot group, combine the requested metrics
        for group in plot_groups:
            # collect series for this group
            series = []  # list of (key, label)
            for metric in group:
                series.extend(metric_to_series(metric))

            if not series:
                continue

            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            plotted_any = False
            for key, label in series:
                vals = [r.get(key) for r in rows_sorted]
                if all(v is None for v in vals):
                    continue
                ax.plot(epochs, vals, label=label)
                plotted_any = True

            if not plotted_any:
                plt.close(fig)

            ax.set_xlabel('epoch')
            ax.set_ylabel(','.join(group))
            ax.legend()
            # filename: join metrics with underscore
            safe_name = '_'.join([g.replace(',', '_') for g in group])
            save_fig(fig, safe_name)

    def register_cli_arguments(self, cli_registry):
        """Register CLI flags for the data exporter.

        Flags are namespaced using the exporter name as a prefix for the dest
        to avoid collisions with other exporters.
        """

        name = 'plot'

        # --plots accepts a comma-separated list
        cli_registry.register(CLIArgument(f"--{name}-plot", config_path=f"exports.plot.plots", action="append", help="Comma-separated list of plots to generate (e.g. loss,accuracy)"))
        cli_registry.register(CLIArgument(f"--{name}-output", config_path=f"exports.plot.output", help="Output folder for exporter"))
        cli_registry.register(CLIArgument(f"--{name}-metadata-dir", config_path=f"exports.plot.metadata_dir", help="The directory to search for checkpoint metadata files"))
