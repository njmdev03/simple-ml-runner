import torch
from pathlib import Path
from ml_runner.core.log_utils import logger
from dataclasses import dataclass
from .base_callback import Callback
from ml_runner.core.config.path_utils import resolve_path_template, ensure_dir

@dataclass
class CheckpointCallback(Callback):
    """
    Handles saving model weights and sets metadata flush targets.
    """
    path: str = "checkpoints/"
    name_template: str = "model_epoch_{epoch}"
    metadata_format: str = "json"
    every_n_epochs: int = 1

    def on_epoch_end(self, engine):
        if engine.train_state.epoch % self.every_n_epochs == 0:
            self._handle_checkpoint(engine)

    def on_eval_end(self, engine):
        # We only set targets if standalone eval or specific epochs.
        # But usually in eval we just want to flush to whatever path corresponds to current epoch.
        self._handle_checkpoint(engine, save_weights=False)

    def _handle_checkpoint(self, engine, save_weights=True):
        epoch = engine.eval_state.epoch if engine.eval_state.epoch > 0 else engine.train_state.epoch

        # Build context for template
        context = {"epoch": epoch}

        base_name = resolve_path_template(self.name_template, context)
        ckpt_filename = base_name if base_name.endswith(".pt") else f"{base_name}.pt"
        ckpt_path = Path(self.path) / ckpt_filename

        # 1. Save weights if requested
        if save_weights:
            ensure_dir(str(ckpt_path))
            torch.save(engine.model.state_dict(), ckpt_path)
            logger.info(f"Saved weights to {ckpt_path}")

        # 2. Update metadata with basic info and SET FLUSH TARGET
        engine.metadata.update("epoch", epoch)
        engine.metadata.update("checkpoint_path", ckpt_filename)

        # Resolving sidecar path
        sidecar_path = ckpt_path.with_suffix(f".{self.metadata_format}") if self.metadata_format != "pt" else ckpt_path
        engine.metadata.set_flush_target(sidecar_path, self.metadata_format)

        # Also add train_loss to metadata if available
        if engine.train_state.batch > 0:
            train_loss = float(engine.train_state.total_loss / engine.train_state.batch)
            engine.metadata.update("train_loss", train_loss)

        # Metrics are usually added by another callback or via eval_state
        if engine.eval_state.batch > 0:
            eval_metrics = {"loss": float(engine.eval_state.total_loss / engine.eval_state.batch)}
            for k, v in engine.eval_state.total_metrics.items():
                eval_metrics[k] = float(v / engine.eval_state.batch)
            engine.metadata.update("eval_metrics", eval_metrics)
