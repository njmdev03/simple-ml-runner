import logging
from pathlib import Path
from typing import List, Union, Dict, Any, Optional, Type

import torch
from torch.utils.data import DataLoader

from ml_runner.core.config import loader as config_loader
from ml_runner.core.config.utils import merge_dicts
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.engine import Engine
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.extensions.extension_manager import ExtensionManager
from ml_runner.core.registries import (
    ModelRegistry,
    DatasetRegistry,
    OptimizerRegistry,
    LossRegistry,
    MetricRegistry,
    TaskRegistry,
)
from ml_runner.core.registries.callbacks import attach
from ml_runner.core.runtime import Runtime
from ml_runner.core.utils.path_utils import resolve_path_template, ensure_dir
from ml_runner.core.log_utils.logger_setup import setup_logging
from ml_runner.core.callbacks.epoch_logger import EpochLogger
from ml_runner.core.callbacks.eval_logger import EvalLogger
from ml_runner.core.callbacks.evaluation_callback import EvaluationCallback

logger = logging.getLogger(__name__)

class JobBuilder:
    """
    The central orchestrator (Composition Root) for building and executing ML jobs.
    Handles configuration loading, extension management, and component assembly.
    """

    def __init__(self, extension_manager: Optional[ExtensionManager] = None):
        self.ext_manager = extension_manager or ExtensionManager()
        self.ext_manager.register_extensions()

    def load_config(self, config_paths: List[str], overrides: Optional[Dict[str, Any]] = None, exporter_config_classes: Optional[Dict[str, Type]] = None) -> RunConfig:
        """
        Loads and merges configuration files and applies CLI overrides.

        Args:
            config_paths: List of paths to YAML/TOML config files.
            overrides: A dictionary of key-path overrides (e.g., {'optimizer.lr': 0.01}).
            exporter_config_classes: Optional mapping of exporter names to their config classes.

        Returns:
            A validated RunConfig instance.
        """
        cfg_dict = {}
        for path in config_paths:
            new_cfg = config_loader.load_config(path)
            cfg_dict = merge_dicts(cfg_dict, new_cfg)

        if overrides:
            cfg_dict = merge_dicts(cfg_dict, overrides)

        # Build RunConfig with extension-specific schemas
        extension_config_classes = self.ext_manager.get_config_classes()

        run_cfg = RunConfig.from_dict(
            cfg_dict,
            extension_config_classes=extension_config_classes,
            exporter_config_classes=exporter_config_classes
        )

        return run_cfg

    def build_runtime(self, run_cfg: RunConfig) -> Runtime:
        """
        Instantiates models, tasks, loaders, and services based on the config.

        Args:
            run_cfg: The validated experiment configuration.

        Returns:
            A Runtime instance containing the assembled experiment state.
        """
        # Device resolution
        device = self._resolve_device(run_cfg.device)

        # Datasets & Loaders
        dataset_cls = DatasetRegistry.get(run_cfg.dataset.name)
        train_ds = dataset_cls(**run_cfg.dataset.params, train=True)
        val_ds = dataset_cls(**run_cfg.dataset.params, train=False)

        dl_cfg = run_cfg.dataloader
        train_loader = DataLoader(
            train_ds,
            batch_size=dl_cfg.batch_size,
            shuffle=dl_cfg.shuffle,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
            drop_last=dl_cfg.drop_last,
            **(dl_cfg.params or {}),
        )

        val_batch_size = getattr(run_cfg.evaluation, "batch_size", dl_cfg.batch_size)
        val_loader = DataLoader(
            val_ds,
            batch_size=val_batch_size,
            num_workers=dl_cfg.num_workers,
            pin_memory=dl_cfg.pin_memory,
            **(dl_cfg.params or {}),
        )

        # Model, Loss, Metrics
        model_cls = ModelRegistry.get(run_cfg.model.name)
        model = model_cls(**run_cfg.model.params).to(device)

        loss_cls = LossRegistry.get(run_cfg.loss.name)
        loss_fn = loss_cls(**run_cfg.loss.params)

        metrics = []
        for m in run_cfg.evaluation.metrics:
            component = MetricRegistry.get(m)
            metrics.append(component() if isinstance(component, type) else component)

        # Task
        task_cls = TaskRegistry.get(run_cfg.task.name)
        task = task_cls(
            model=model,
            loss_fn=loss_fn,
            optimizer=None, # Will be set below
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            metrics=metrics,
            **(run_cfg.task.params or {})
        )

        # Optimizer
        opt_cfg = run_cfg.optimizer
        optimizer_cls = OptimizerRegistry.get(opt_cfg.name)
        optimizer = optimizer_cls(model.parameters(), lr=opt_cfg.lr, **opt_cfg.params)
        task.optimizer = optimizer

        # Assemble Runtime
        runtime = Runtime(
            model=model,
            optimizer=optimizer,
            task=task,
            train_loader=train_loader,
            val_loader=val_loader,
            device=str(device),
            metrics=metrics,
            config=run_cfg
        )

        # Configure and Build Extensions
        self.ext_manager.configure(run_cfg)
        self.ext_manager.build(runtime)

        return runtime

    def build_engine(self, runtime: Runtime, event_manager: Optional[EventManager] = None) -> Engine:
        """
        Creates the execution Engine and wires it to extensions and standard callbacks.

        Args:
            runtime: The assembled experiment runtime.
            event_manager: Optional custom event manager.

        Returns:
            An Engine instance ready for execution.
        """
        event_manager = event_manager or EventManager()

        # Register standard callbacks
        attach(EpochLogger(), event_manager)
        attach(EvalLogger(), event_manager)

        if runtime.config.evaluation.eval_during_training:
            attach(EvaluationCallback(every_n_epochs=runtime.config.evaluation.eval_frequency), event_manager)

        # Create Engine
        engine = Engine(task=runtime.task, event_manager=event_manager)

        # Attach Extensions
        self.ext_manager.attach(engine)

        return engine

    def setup_logging(self, run_cfg: RunConfig):
        """
        Configures logging based on the experiment configuration.
        """
        context = {"experiment_name": run_cfg.experiment.name, "device": run_cfg.device}
        log_file_name = resolve_path_template(run_cfg.logging.log_file, context)
        log_dir = resolve_path_template(run_cfg.logging.output_dir, context)
        log_path = str(Path(log_dir) / log_file_name)
        ensure_dir(log_path)

        # Clear existing handlers
        root_logger = logging.getLogger("mltool")
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        setup_logging(level=run_cfg.logging.level, log_file=log_path)
        logger.info(f"Initialized Job: {run_cfg.experiment.name}")

    def _resolve_device(self, device_spec: Union[str, List[str]]) -> torch.device:
        if isinstance(device_spec, str):
            device_spec = [device_spec]

        for d in device_spec:
            if d == "auto":
                d = "cuda" if torch.cuda.is_available() else "cpu"
            try:
                device = torch.device(d)
                # Test if device is actually available
                if device.type == "cuda" and not torch.cuda.is_available():
                    continue
                return device
            except Exception:
                continue

        return torch.device("cpu")
