import torch
from pathlib import Path
from dataclasses import dataclass

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.registries.callbacks import Callback
from ml_runner.core.engine.engine import EngineEvent
from ml_runner.core.utils.path_utils import resolve_path_template, ensure_dir
from ml_runner.core.log_utils import logger
from ml_runner.core.cli_interface.cli_registry import CLIRegistry
from ml_runner.core.cli_interface.cli_argument import CLIArgument


@dataclass
class Config:
    enabled: bool = True
    directory: str = "checkpoints/{experiment_name}"
    name_template: str = "ckpt_epoch_{epoch}.pt"
    metadata_format: str = "json"
    frequency: int = 1  # every N epochs


@Extension("checkpoints", config_class=Config)
class CheckpointExtension(BaseExtension):

    def setup(self, event_manager: EventManager, global_config: RunConfig, config: Config = None):
        if not config or not config.enabled or config.frequency <= 0:
            return

        context = {
            "experiment_name": global_config.experiment.name,
            "device": global_config.device
        }

        self.path = resolve_path_template(config.directory, context)
        self.name_template = config.name_template
        self.metadata_format = config.metadata_format
        self.every_n_epochs = config.frequency
        ensure_dir(self.path)
        # attach(self, event_manager)

        CLIRegistry.register(CLIArgument("--checkpoint-dir", help="Directory for saving checkpoints",
                             config_path="extensions.checkpoint.directory"))
        CLIRegistry.register(CLIArgument("--checkpoint-frequency", type=int, help="How often to save checkpoints",
                             config_path="extensions.checkpoint.frequency"))

    @Callback(EngineEvent.EPOCH_END)
    def on_epoch_end(self, engine, **kwargs):
        if engine.train_state.epoch % self.every_n_epochs == 0:
            self._handle_checkpoint(engine)

    @Callback(EngineEvent.EVAL_END)
    def on_eval_end(self, engine, **kwargs):
        self._handle_checkpoint(engine, save_weights=False)

    def _handle_checkpoint(self, engine, save_weights=True):
        epoch = engine.eval_state.epoch if engine.eval_state.epoch > 0 else engine.train_state.epoch

        context = {"epoch": epoch}
        base_name = resolve_path_template(self.name_template, context)
        ckpt_filename = base_name if base_name.endswith(".pt") else f"{base_name}.pt"
        ckpt_path = Path(self.path) / ckpt_filename

        if save_weights:
            ensure_dir(str(ckpt_path))
            torch.save(engine.model.state_dict(), ckpt_path)
            logger.info(f"Saved weights to {ckpt_path}")

        engine.metadata.update("epoch", epoch)
        engine.metadata.update("checkpoint_path", ckpt_filename)

        sidecar_path = ckpt_path.with_suffix(f".{self.metadata_format}") if self.metadata_format != "pt" else ckpt_path
        engine.metadata.set_flush_target(sidecar_path, self.metadata_format)

        if engine.train_state.batch > 0:
            train_loss = float(engine.train_state.total_loss / engine.train_state.batch)
            engine.metadata.update("train_loss", train_loss)

        if engine.eval_state.batch > 0:
            eval_metrics = {"loss": float(engine.eval_state.total_loss / engine.eval_state.batch)}
            for k, v in engine.eval_state.total_metrics.items():
                eval_metrics[k] = float(v / engine.eval_state.batch)
            engine.metadata.update("eval_metrics", eval_metrics)
