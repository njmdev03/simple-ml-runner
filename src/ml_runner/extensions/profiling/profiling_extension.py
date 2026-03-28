from dataclasses import dataclass
from typing import Dict, Optional

from ml_runner.core.extensions.base_extension import BaseExtension
from ml_runner.core.registries import Extension
from ml_runner.core.config.schema import RunConfig
from ml_runner.core.engine.event_manager import EventManager
from ml_runner.core.engine.engine import EngineEvent
from ml_runner.extensions.profiling.timer import Timer


@dataclass
class ProfilingConfig:
    enabled: bool = False


@Extension("profiling", config_class=ProfilingConfig)
class ProfilingExtension(BaseExtension):
    """
    Extension that provides enhanced timing and profiling capabilities.
    Measures: whole job time, training time, evaluation time, and per-epoch timings.
    """
    def setup(self, event_manager: EventManager, global_config: RunConfig, config: Optional[ProfilingConfig] = None):
        if not config or not config.enabled:
            return

        self.timers: Dict[str, Timer] = {
            "job": Timer(),
            "train_total": Timer(),
            "eval_total": Timer(),
            "epoch_total": Timer(),
            "epoch_train": Timer(),
            "epoch_eval": Timer()
        }

        event_manager.subscribe(self.on_job_start, EngineEvent.JOB_START)
        event_manager.subscribe(self.on_job_end, EngineEvent.JOB_END)
        event_manager.subscribe(self.on_train_start, EngineEvent.TRAIN_START)
        event_manager.subscribe(self.on_train_end, EngineEvent.TRAIN_END)
        event_manager.subscribe(self.on_epoch_start, EngineEvent.EPOCH_START)
        event_manager.subscribe(self.on_epoch_end, EngineEvent.EPOCH_END)
        event_manager.subscribe(self.on_eval_start, EngineEvent.EVAL_START)
        event_manager.subscribe(self.on_eval_end, EngineEvent.EVAL_END)

    def on_job_start(self, engine, **kwargs):
        self.timers["job"].start()

    def on_job_end(self, engine, **kwargs):
        self.timers["job"].pause()
        self._record_final_times(engine)

    def on_train_start(self, engine, **kwargs):
        self.timers["train_total"].start()

    def on_train_end(self, engine, **kwargs):
        self.timers["train_total"].pause()

    def on_epoch_start(self, engine, **kwargs):
        self.timers["epoch_total"].reset()
        self.timers["epoch_train"].reset()
        self.timers["epoch_eval"].reset()
        self.timers["epoch_total"].start()
        self.timers["epoch_train"].start()

    def on_epoch_end(self, engine, **kwargs):
        self.timers["epoch_train"].pause()
        self.timers["epoch_total"].pause()
        self._record_epoch_times(engine)

    def on_eval_start(self, engine, **kwargs):
        self.timers["epoch_train"].pause()
        self.timers["train_total"].pause()
        self.timers["epoch_eval"].start()
        self.timers["eval_total"].start()
        self.timers["job"].start()
        self.timers["epoch_total"].start()

    def on_eval_end(self, engine, **kwargs):
        self.timers["epoch_eval"].pause()
        self.timers["eval_total"].pause()
        if engine.train_state.epoch > 0:
            self.timers["epoch_train"].start()
            self.timers["train_total"].start()
        else:
            self.timers["job"].pause()

    def _record_epoch_times(self, engine):
        durations = {
            "epoch_duration": self.timers["epoch_total"].elapsed,
            "epoch_train_duration": self.timers["epoch_train"].elapsed,
            "epoch_eval_duration": self.timers["epoch_eval"].elapsed,
        }
        for k, v in durations.items():
            engine.trigger("record_metric", key=k, value=v, namespace="timing")

    def _record_final_times(self, engine):
        durations = {
            "job_time_total": self.timers["job"].elapsed,
            "train_time_total": self.timers["train_total"].elapsed,
            "eval_time_total": self.timers["eval_total"].elapsed,
        }
        for k, v in durations.items():
            engine.trigger("record_metric", key=k, value=v, namespace="timing")
