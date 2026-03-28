import os
import glob
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from ml_runner.core.utils.path_utils import resolve_path_template, ensure_dir
from ml_runner.core.log_utils.logger_setup import setup_logging
from ml_runner.core.callbacks.epoch_logger import EpochLogger
from ml_runner.core.callbacks.eval_logger import EvalLogger
from ml_runner.core.callbacks.evaluation_callback import EvaluationCallback
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.registries.callbacks import attach

from ml_runner.core.engine.engine import Engine, EngineEvent
from ml_runner.core.registries import (
    ModelRegistry,
    DatasetRegistry,
    OptimizerRegistry,
    LossRegistry,
    MetricRegistry,
    TaskRegistry,
)


# -------------------------
# Runtime Builder
# -------------------------
def build_runtime(run_cfg):
    """
    Constructs model, task, optimizer, loss, metrics, dataloaders.
    Returns: task instance, device
    Assumes run_cfg has a strict schema with all required attributes.
    """

    # Device: first available from list
    if isinstance(run_cfg.device, list):
        for d in run_cfg.device:
            if d == "auto":
                d = "cuda" if torch.cuda.is_available() else "cpu"
            if d == "cuda" and torch.cuda.is_available():
                device = "cuda"
                break
            elif d == "cpu":
                device = "cpu"
                break
        else:
            raise RuntimeError(f"No available devices in {run_cfg.device}")
    else:
        # Single device string
        if run_cfg.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = run_cfg.device

    # -------------------------
    # Dataset & DataLoaders
    # -------------------------
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

    # Allow overriding val/eval batch size separately
    val_batch_size = getattr(run_cfg.evaluation, "batch_size", None) if hasattr(run_cfg, "evaluation") else None
    if val_batch_size is None:
        val_batch_size = dl_cfg.batch_size

    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        num_workers=dl_cfg.num_workers,
        pin_memory=dl_cfg.pin_memory,
        **(dl_cfg.params or {}),
    )

    # -------------------------
    # Model, Loss, Optimizer
    # -------------------------
    model_cls = ModelRegistry.get(run_cfg.model.name)
    model = model_cls(**run_cfg.model.params)

    loss_cls = LossRegistry.get(run_cfg.loss.name)
    loss_fn = loss_cls(**run_cfg.loss.params)

    # -------------------------
    # Metrics
    # -------------------------
    metrics = []
    for m in run_cfg.evaluation.metrics:
        component = MetricRegistry.get(m)
        if isinstance(component, type):
            # It's a class, instantiate it once for the job
            metrics.append(component())
        else:
            # It's already a callable function or object
            metrics.append(component)

    # -------------------------
    # Task
    # -------------------------
    task_type = run_cfg.task.name
    TaskClass = TaskRegistry.get(task_type)

    task = TaskClass(
        model=model,
        loss_fn=loss_fn,
        optimizer=None,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        metrics=metrics,
        **(run_cfg.task.params or {})
    )

    # Move model to device (Task should have ensured it is on the correct device)
    task.model.to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer_cls = OptimizerRegistry.get(run_cfg.optimizer.name)
    optimizer = optimizer_cls(task.model.parameters(), lr=run_cfg.optimizer.lr, **run_cfg.optimizer.params)
    task.optimizer = optimizer

    return task, device

# -------------------------
# Job Execution
# -------------------------
def run_job(run_cfg, args, ext_manager):
    """
    Sets up logging, builds runtime, constructs callbacks, and executes training/evaluation.
    Assumes all attributes exist due to strict schema.
    """

    import logging

    # -------------------------
    # Path resolution
    # -------------------------
    context = {"experiment_name": run_cfg.experiment.name, "device": run_cfg.device}

    log_file_name = resolve_path_template(run_cfg.logging.log_file, context)
    log_dir = resolve_path_template(run_cfg.logging.output_dir, context)
    log_path = str(Path(log_dir) / log_file_name)
    ensure_dir(log_path)

    # Reset logging handlers
    root_logger = logging.getLogger("mltool")
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Setup logging
    job_logger = setup_logging(level=run_cfg.logging.level, log_file=log_path)
    job_logger.info(f"Starting Job: {run_cfg.experiment.name}")

    # -------------------------
    # Build Task
    # -------------------------
    task, device = build_runtime(run_cfg)

    # -------------------------
    # Callbacks & Event Management
    # -------------------------
    event_manager = EventManager()

    # Register standard loggers
    attach(EpochLogger(), event_manager)
    attach(EvalLogger(), event_manager)

    if run_cfg.evaluation.eval_during_training:
        attach(EvaluationCallback(every_n_epochs=run_cfg.evaluation.eval_frequency), event_manager)

    # Setup extensions
    ext_manager.before_job(config=run_cfg, event_manager=event_manager)

    # -------------------------
    # Engine
    # -------------------------
    engine = Engine(task=task, event_manager=event_manager)

    engine._trigger(EngineEvent.JOB_START)

    # -------------------------
    # Training
    # -------------------------
    if run_cfg.training.enabled:
        job_logger.info(f"Training for {run_cfg.training.epochs} epochs")
        engine.train(epochs=run_cfg.training.epochs)

    # -------------------------
    # Evaluation
    # -------------------------
    # if run_cfg.evaluation.enabled:
    #     if run_cfg.evaluation.eval_checkpoints and not run_cfg.training.enabled:
    #         # Evaluate all checkpoints
    #         checkpoint_cfg = run_cfg.extensions.get("checkpoints")
    #         if not checkpoint_cfg.enabled:
    #             job_logger.warning("No checkpoints extension configured, running default evaluation")
    #             engine.evaluate()
    #         else:
    #             checkpoint_dir = resolve_path_template(checkpoint_cfg.directory, context)
    #             checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.pt"))
    #             if not checkpoints:
    #                 job_logger.warning(f"No checkpoints found in {checkpoint_dir}")
    #                 engine.evaluate()
    #             else:
    #                 job_logger.info(f"Found {len(checkpoints)} checkpoints")
    #                 for ckpt in sorted(checkpoints):
    #                     job_logger.info(f"Evaluating checkpoint: {ckpt}")
    #                     task.load_checkpoint(ckpt)

    #                     import re
    #                     match = re.search(r"epoch_(\d+)", Path(ckpt).name)
    #                     if match:
    #                         engine.eval_state.epoch = int(match.group(1))

    #                     engine.evaluate()
    #     else:
    #         job_logger.info("Running standard evaluation")
    #         engine.evaluate()

    engine._trigger(EngineEvent.JOB_END)

    job_logger.info(f"Job {run_cfg.experiment.name} Complete")
