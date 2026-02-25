import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class Visualizer:
    """Generates visualizations for ML training results and profiling data."""

    def __init__(self, output_dir: str = "vis", format: str = "png"):
        self.output_dir = output_dir
        self.format = format.lower()
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

    def _save_or_show(self, fig, filename: str, show: bool = False):
        """Save figure to file or display interactively."""
        if filename:
            path = os.path.join(self.output_dir, f"{filename}.{self.format}")
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved: {path}")

        # If showing, we leave the figure open so plt.show() can be called later
        # If not showing, we close to free memory
        if not show:
            plt.close(fig)

    def _apply_style(self, ax):
        """Apply consistent styling to axes."""
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, alpha=0.3)

    def _set_accessible_cycler(self, ax):
        """Set a colorblind-friendly property cycler for the given axes."""
        from cycler import cycler
        # Tableau colorblind-friendly palette (10 colors)
        colors = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1',
                  '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
        markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', 'h', '+']
        lines = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--'] # Pad to 10

        custom_cycler = (cycler(color=colors) +
                        cycler(marker=markers) +
                        cycler(linestyle=lines))
        ax.set_prop_cycle(custom_cycler)

    # =========================================================================
    # Results CSV Visualizations
    # =========================================================================

    def plot_loss_accuracy(self, results_df: pd.DataFrame, show: bool = False,
                           datasets: List[str] = None, metrics: List[str] = None,
                           ax: plt.Axes = None):
        """Combined loss + accuracy over epochs with dual y-axes.

        Args:
            datasets: Filter by 'Training', 'Testing', or both. None = all.
            metrics: Filter by 'loss', 'accuracy', or both. None = all.
            ax: Optional matplotlib axes to plot on. If provided, no new figure is created.
        """
        if datasets is None:
            datasets = ['Training', 'Testing']
        if metrics is None:
            metrics = ['loss', 'accuracy']

        save_plot = ax is None
        # If external ax is provided, use it primarily for loss (left axis)
        if ax is None:
            fig, ax1 = plt.subplots(figsize=(10, 6))
        else:
            fig, ax1 = ax.figure, ax

        # Get loss and metric columns
        loss_cols = [c for c in results_df.columns if 'loss' in c.lower()]
        
        metric_cols = []
        for c in results_df.columns:
            if c.lower() in ('epoch', 'dataset', 'source') or 'loss' in c.lower():
                continue
            if metrics is None or c.lower() in [m.lower() for m in metrics]:
                metric_cols.append(c)

        show_loss = loss_cols and (metrics is None or 'loss' in [m.lower() for m in metrics])
        show_metric = bool(metric_cols)

        if not show_loss and not show_metric:
            logger.warning("No matching metrics found in results.")
            if save_plot:
                plt.close(fig)
            return

        loss_col = loss_cols[0] if loss_cols else None

        # Plot loss on left y-axis
        ax1.set_xlabel('Epoch', fontsize=12)
        if show_loss:
            ax1.set_ylabel('Loss', fontsize=12, color='#e74c3c')
            self._set_accessible_cycler(ax1)

            for dataset_name in datasets:
                subset = results_df[results_df['dataset'] == dataset_name]
                if not subset.empty:
                    ax1.plot(subset['epoch'], subset[loss_col],
                            alpha=0.7, label=f'{dataset_name} Loss')

        ax1.tick_params(axis='y', labelcolor='#e74c3c' if show_loss else 'black')
        self._apply_style(ax1)

        # Plot metrics on right y-axis
        if show_metric:
            ax2 = ax1.twinx()
            ax2.set_ylabel('Metrics', fontsize=12, color='#9b59b6')
            self._set_accessible_cycler(ax2)

            for acc_col in metric_cols:
                for dataset_name in datasets:
                    subset = results_df[results_df['dataset'] == dataset_name]
                    if not subset.empty:
                        ax2.plot(subset['epoch'], subset[acc_col],
                                linewidth=2, label=f'{dataset_name} {acc_col}')

            ax2.tick_params(axis='y', labelcolor='#9b59b6')
            ax2.set_ylim(0, 1.05)

        # Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        if show_metric:
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        else:
            ax1.legend(lines1, labels1, loc='center right')

        fig.suptitle('Loss & Metrics Over Epochs', fontsize=14, fontweight='bold')
        fig.tight_layout()

        if save_plot:
            self._save_or_show(fig, 'loss_metrics_combined', show)

    def plot_loss(self, results_df: pd.DataFrame, show: bool = False,
                  datasets: List[str] = None, ax: plt.Axes = None):
        """Loss over epochs with maximized style variety via property cyclers.

        Args:
            datasets: Filter by 'Training', 'Testing', or both. None = all.
            ax: Optional matplotlib axes to plot on.
        """
        if datasets is None:
            datasets = ['Training', 'Testing']

        save_plot = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        loss_cols = [c for c in results_df.columns if 'loss' in c.lower()]
        if not loss_cols:
            logger.warning("No loss columns found.")
            if save_plot:
                plt.close(fig)
            return

        self._set_accessible_cycler(ax)

        for loss_col in loss_cols:
            for dataset_name in datasets:
                subset = results_df[results_df['dataset'] == dataset_name]
                if not subset.empty:
                    label_prefix = 'Train' if dataset_name == 'Training' else 'Test'
                    ax.plot(subset['epoch'], subset[loss_col],
                           linewidth=2, markersize=6, label=f'{label_prefix}: {loss_col}')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Loss Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self._apply_style(ax)

        if save_plot:
            fig.tight_layout()
            self._save_or_show(fig, 'loss', show)

    def plot_metrics(self, results_df: pd.DataFrame, show: bool = False,
                      datasets: List[str] = None, metrics: List[str] = None, ax: plt.Axes = None):
        """Metrics over epochs with maximized style variety via property cyclers.

        Args:
            datasets: Filter by 'Training', 'Testing', or both. None = all.
            ax: Optional matplotlib axes to plot on.
        """
        if datasets is None:
            datasets = ['Training', 'Testing']

        save_plot = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        metric_cols = []
        for c in results_df.columns:
            if c.lower() in ('epoch', 'dataset', 'source') or 'loss' in c.lower():
                continue
            if metrics is None or c.lower() in [m.lower() for m in metrics]:
                metric_cols.append(c)

        if not metric_cols:
            logger.warning("No metric columns found.")
            if save_plot:
                plt.close(fig)
            return

        self._set_accessible_cycler(ax)

        for col in metric_cols:
            for dataset_name in datasets:
                subset = results_df[results_df['dataset'] == dataset_name]
                if not subset.empty:
                    label_prefix = 'Train' if dataset_name == 'Training' else 'Test'
                    ax.plot(subset['epoch'], subset[col],
                           linewidth=2, markersize=6, label=f'{label_prefix}: {col}')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Metrics', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_title('Metrics Over Epochs', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        self._apply_style(ax)

        fig.tight_layout()
        if save_plot:
            self._save_or_show(fig, 'metrics', show)

    # =========================================================================
    # Profiler CSV Visualizations
    # =========================================================================

        ax.set_title('Process Duration Breakdown', fontsize=14, fontweight='bold', pad=20)

        fig.tight_layout()
        if ax is None or fig.get_axes()[0] == ax: # This is a bit hacky, cleaner way below
             pass 

    def plot_duration_table(self, profile_df: pd.DataFrame, show: bool = False,
                            ax: plt.Axes = None):
        """Phase durations displayed as a clean table."""
        # Normalize to long format if needed
        if 'metric' not in profile_df.columns and len(profile_df) == 1:
            profile_df = profile_df.melt(var_name='metric', value_name='value')

        # Filter for high-level durations
        high_level = profile_df[~profile_df['metric'].str.contains('epoch', case=False)].copy()

        if high_level.empty:
            logger.warning("No high-level duration metrics found.")
            return

        # Clean up metric names for display
        high_level['metric'] = high_level['metric'].str.replace('_duration', '').str.replace('_', ' ').str.title()

        save_plot = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, len(high_level) * 0.5 + 1))
        else:
            fig = ax.figure

        ax.axis('off')

        # Prepare table data
        table_data = high_level[['metric', 'value']].values
        table_data = [[m, f"{v:.4f}s"] for m, v in table_data]

        table = ax.table(cellText=table_data, colLabels=['Phase', 'Duration'],
                        loc='center', cellLoc='left', colWidths=[0.6, 0.3])

        # Styling the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.5)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#34495e')
            else:
                cell.set_facecolor('#f2f2f2' if row % 2 == 0 else 'white')

        ax.set_title('Process Duration Breakdown', fontsize=14, fontweight='bold', pad=20)

        fig.tight_layout()
        if save_plot:
            self._save_or_show(fig, 'duration_breakdown', show)

    def plot_epoch_timing(self, profile_df: pd.DataFrame, show: bool = False,
                          ax: plt.Axes = None):
        """Line chart of per-epoch training/testing duration."""
        # Normalize to long format if needed
        if 'metric' not in profile_df.columns and len(profile_df) == 1:
            profile_df = profile_df.melt(var_name='metric', value_name='value')

        save_plot = ax is None
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        self._set_accessible_cycler(ax)

        # Parse epoch timing metrics
        training_times = {}
        testing_times = {}

        for _, row in profile_df.iterrows():
            metric = row['metric']
            value = row['value']

            if 'epoch' in metric.lower():
                # Extract epoch number
                parts = metric.split('_')
                if 'eval' in metric.lower() and 'testing' in metric.lower():
                    # Format: eval_epoch_X_testing
                    try:
                        # Find index of 'epoch'
                        epoch_idx = parts.index('epoch')
                        epoch = int(parts[epoch_idx + 1])
                        testing_times[epoch] = value
                    except (ValueError, IndexError):
                        pass
                elif metric.startswith('epoch_'):
                    # Format: epoch_X (Training time)
                    try:
                        epoch = int(parts[1])
                        training_times[epoch] = value
                    except (ValueError, IndexError):
                        pass

        if not training_times and not testing_times:
            logger.warning("No per-epoch timing data found.")
            if ax is None:
                plt.close(fig)
            return

        if training_times:
            epochs = sorted(training_times.keys())
            values = [training_times[e] for e in epochs]
            ax.plot(epochs, values, linewidth=2, label='Training')

        if testing_times:
            epochs = sorted(testing_times.keys())
            values = [testing_times[e] for e in epochs]
            ax.plot(epochs, values, linewidth=2, label='Testing')

        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Duration (seconds)', fontsize=12)
        ax.set_title('Per-Epoch Duration', fontsize=14, fontweight='bold')
        ax.legend()
        self._apply_style(ax)

        fig.tight_layout()
        if save_plot:
            self._save_or_show(fig, 'epoch_timing', show)

    # =========================================================================
    # Model Visualizations
    # =========================================================================

    def plot_sample_predictions(self, model, dataset, device,
                                 num_samples: int = 9, show: bool = False,
                                 ax: plt.Axes = None):
        """Grid of dataset samples with actual vs predicted labels."""
        if num_samples <= 0:
            logger.info("Skipping sample predictions (num_samples <= 0)")
            return

        import torch

        model.eval()

        save_plot = ax is None
        if ax is not None:
             indices = np.random.choice(len(dataset), 1, replace=False)
             axes = [ax]
             fig = ax.figure
        else:
            # Determine grid size
            grid_size = int(np.ceil(np.sqrt(num_samples)))
            fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
            axes = axes.flatten() if num_samples > 1 else [axes]

            # Get random samples
            indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

        with torch.no_grad():
            for idx, ax in enumerate(axes):
                if idx >= len(indices):
                    ax.axis('off')
                    continue

                sample_idx = indices[idx]
                data, target = dataset[sample_idx]

                # Add batch dimension for model
                if isinstance(data, torch.Tensor):
                    input_data = data.unsqueeze(0).to(device)
                else:
                    input_data = torch.tensor(data).unsqueeze(0).to(device)

                # Get prediction
                output = model(input_data)
                pred = output.argmax(dim=1).item()
                actual = target if isinstance(target, int) else target.item()

                # Display image if applicable
                if isinstance(data, torch.Tensor) and len(data.shape) >= 2:
                    # Handle image data
                    img = data.cpu().numpy()
                    if len(img.shape) == 3:
                        if img.shape[0] in [1, 3]:  # CHW format
                            img = np.transpose(img, (1, 2, 0))
                        if img.shape[2] == 1:
                            img = img.squeeze(2)

                    ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
                else:
                    # For non-image data, show as text
                    ax.text(0.5, 0.5, f'Sample {sample_idx}',
                           ha='center', va='center', fontsize=10)

                # Color title based on correctness
                color = '#2ecc71' if pred == actual else '#e74c3c'
                ax.set_title(f'Pred: {pred} | Actual: {actual}',
                            fontsize=10, color=color, fontweight='bold')
                ax.axis('off')

        if save_plot:
            fig.suptitle('Sample Predictions', fontsize=14, fontweight='bold')
            fig.tight_layout()
            self._save_or_show(fig, 'sample_predictions', show)

    def plot_model_architecture(self, model, input_shape: Tuple = None, show: bool = False,
                                ax: plt.Axes = None):
        """Visualize model architecture in a portrait layout with dynamic height."""
        import torch.nn as nn

        # Extract layers
        layers = []
        for name, module in model.named_modules():
            if name == '':
                continue
            layer_type = module.__class__.__name__

            details = ''
            if isinstance(module, nn.Linear):
                details = f'({module.in_features} → {module.out_features})'
            elif isinstance(module, nn.Conv2d):
                details = f'({module.in_channels}→{module.out_channels}, k={module.kernel_size})'
            elif isinstance(module, nn.Dropout):
                details = f'(p={module.p})'
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                details = f'({module.num_features})'

            layers.append({'type': layer_type, 'details': details})

        if not layers:
            return

        # Dynamic figure size based on layers (Portrait orientation)
        num_layers = len(layers)

        save_plot = ax is None
        if ax is None:
            fig_height = max(6, num_layers * 0.8)
            fig, ax = plt.subplots(figsize=(6, fig_height))
        else:
            fig = ax.figure

        ax.axis('off')

        box_width = 0.8
        box_height = 0.05 / (num_layers / 10) if num_layers > 10 else 0.05
        # Simplified coordinate system for drawing
        y_coords = np.linspace(0.95, 0.05, num_layers)

        color_map = {
            'Linear': '#3498db', 'Conv2d': '#2ecc71', 'ReLU': '#e74c3c',
            'Dropout': '#9b59b6', 'BatchNorm2d': '#1abc9c', 'MaxPool2d': '#34495e',
            'Flatten': '#95a5a6', 'LogSoftmax': '#c0392b'
        }

        for i, (y, layer) in enumerate(zip(y_coords, layers)):
            color = color_map.get(layer['type'], '#bdc3c7')

            # Box centered at 0.5
            rect = mpatches.FancyBboxPatch(
                (0.1, y - 0.02), 0.8, 0.04,
                boxstyle="round,pad=0.01", facecolor=color, edgecolor='black',
                alpha=0.9, linewidth=1
            )
            ax.add_patch(rect)

            text = f"{layer['type']}\n{layer['details']}" if layer['details'] else layer['type']
            ax.text(0.5, y, text, ha='center', va='center', fontsize=9,
                   fontweight='bold', color='white')

            if i < num_layers - 1:
                next_y = y_coords[i+1]
                ax.annotate('', xy=(0.5, next_y + 0.02), xytext=(0.5, y - 0.02),
                           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Architecture: {model.__class__.__name__}', fontsize=12, fontweight='bold')

        if save_plot:
            fig.tight_layout()
            self._save_or_show(fig, 'model_architecture', show)

    def generate_grid_view(self, results_df: pd.DataFrame = None,
                           profile_df: pd.DataFrame = None, show: bool = False,
                           datasets: List[str] = None, metrics: List[str] = None):
        """Generate a single figure with all available plots in a grid."""
        plots = []

        if results_df is not None:
            plots.append(('loss_acc', lambda ax: self.plot_loss_accuracy(results_df, False, datasets, metrics, ax)))
            if metrics is None or 'loss' in [m.lower() for m in metrics]:
                plots.append(('loss', lambda ax: self.plot_loss(results_df, False, datasets, ax)))
            if metrics is None or any(m.lower() != 'loss' for m in metrics):
                plots.append(('metrics', lambda ax: self.plot_metrics(results_df, False, datasets, metrics, ax)))

        if profile_df is not None:
             plots.append(('duration', lambda ax: self.plot_duration_table(profile_df, False, ax)))
             plots.append(('timing', lambda ax: self.plot_epoch_timing(profile_df, False, ax)))

        if not plots:
            logger.warning("No data available for grid view.")
            return

        # Calculate grid dimensions
        n_plots = len(plots)
        cols = 3 if n_plots >= 3 else n_plots
        rows = int(np.ceil(n_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        if n_plots > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        # Plot each item
        for i, (name, plot_func) in enumerate(plots):
            plot_func(axes[i])

        # Hide empty subplots
        for i in range(len(plots), len(axes)):
            axes[i].axis('off')

        fig.tight_layout()
        self._save_or_show(fig, 'grid_view', show)

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def load_results_csv(path: str) -> pd.DataFrame:
        """Load and parse results CSV file."""
        return pd.read_csv(path)

    @staticmethod
    def load_profile_csv(path: str) -> pd.DataFrame:
        """Load and parse profile CSV file."""
        return pd.read_csv(path)

    def generate_all(self, results_df: pd.DataFrame = None,
                     profile_df: pd.DataFrame = None, show: bool = False,
                     datasets: List[str] = None, metrics: List[str] = None,
                     layout: str = 'individual'):
        """Generate all applicable visualizations.

        Args:
            datasets: Filter by 'Training', 'Testing', or both. None = all.
            metrics: Filter by 'loss', 'accuracy', or both. None = all.
            layout: 'individual' (separate windows) or 'grid' (single window).
        """
        if layout == 'grid':
            self.generate_grid_view(results_df, profile_df, show, datasets, metrics)
            return

        if results_df is not None:
            self.plot_loss_accuracy(results_df, show, datasets=datasets, metrics=metrics)
            if metrics is None or 'loss' in [m.lower() for m in metrics]:
                self.plot_loss(results_df, show, datasets=datasets)
            if metrics is None or any(m.lower() != 'loss' for m in metrics):
                self.plot_metrics(results_df, show, datasets=datasets, metrics=metrics)

        if profile_df is not None:
            self.plot_duration_table(profile_df, show)
            self.plot_epoch_timing(profile_df, show)
