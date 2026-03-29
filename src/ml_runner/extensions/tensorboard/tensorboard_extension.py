from dataclasses import dataclass
from typing import Optional
import os
import shutil

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.engine.engine import Engine, EngineEvent
from ml_runner.core.utils.path_utils import resolve_path_template, ensure_dir

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


@dataclass
class TensorboardConfig:
    enabled: bool = False
    log_dir: str = "tensorboard/{experiment_name}"


@Extension("tensorboard", config_class=TensorboardConfig)
class TensorboardExtension(BaseExtension):
    """
    Extension that integrates TensorBoard logging directly.
    """
    def setup(self, event_manager: EventManager, global_config: RunConfig, config: Optional[TensorboardConfig] = None):
        if config and config.enabled:
            context = {
                "experiment_name": global_config.experiment.name,
                "device": global_config.device
            }
            self.log_dir = resolve_path_template(config.log_dir, context)
            self.writer = None
            self._last_epoch = 0

            event_manager.subscribe(self.on_job_start, EngineEvent.JOB_START)
            event_manager.subscribe(self.on_epoch_end, EngineEvent.EPOCH_END)
            event_manager.subscribe(self.on_eval_end, EngineEvent.EVAL_END)
            event_manager.subscribe(self.on_train_end, EngineEvent.TRAIN_END)
            event_manager.subscribe(self.on_job_end, EngineEvent.JOB_END)
            event_manager.subscribe(self.on_record_metric, "record_metric")

    def on_job_start(self, engine, **kwargs):
        if SummaryWriter and not self.writer:
            # Clear existing traces if the directory is already populated
            if os.path.exists(self.log_dir):
                shutil.rmtree(self.log_dir, ignore_errors=True)
            ensure_dir(self.log_dir)

            self.writer = SummaryWriter(log_dir=self.log_dir)
        self._last_epoch = 0

    def on_epoch_end(self, engine, **kwargs):
        if not self.writer: return
        epoch = engine.train_state.epoch
        self._last_epoch = epoch

        if engine.train_state.batch > 0:
            loss = engine.train_state.total_loss / engine.train_state.batch
            self.writer.add_scalar("Train/Loss", loss, epoch)

        timing = engine.metadata.get("timing", {})
        if "epoch_duration" in timing:
            self.writer.add_scalar("Time/Epoch", timing["epoch_duration"], epoch)
        if "epoch_train_duration" in timing:
            self.writer.add_scalar("Time/EpochTrain", timing["epoch_train_duration"], epoch)
        if "epoch_eval_duration" in timing:
            self.writer.add_scalar("Time/EpochEval", timing["epoch_eval_duration"], epoch)

    def on_eval_end(self, engine: Engine, **kwargs):
        if not self.writer: return
        epoch = engine.eval_state.epoch
        self._last_epoch = epoch

        if engine.eval_state.batch > 0:
            loss = engine.eval_state.total_loss / engine.eval_state.batch
            self.writer.add_scalar("Eval/Loss", loss, epoch)

            # Log all aggregated metrics
            for k, v in engine.eval_state.total_metrics.items():
                if v is not None:
                    self.writer.add_scalar(f"Eval/{k}", float(v / engine.eval_state.batch), epoch)

    def on_record_metric(self, key, value, engine=None, namespace=None, **kwargs):
        if not self.writer:
            return

        if engine:
            epoch = engine.train_state.epoch if engine.train_state.epoch > 0 else engine.eval_state.epoch
            self._last_epoch = epoch
        else:
            epoch = self._last_epoch

        if namespace:
            tag = f"{namespace}/{key}"
        else:
            tag = key

        self.writer.add_scalar(tag, value, epoch)

    def on_train_end(self, engine, **kwargs):
        if self.writer:
            self.writer.close()
            self.writer = None

    def on_job_end(self, engine, **kwargs):
        if self.writer:
            self.writer.close()
            self.writer = None
