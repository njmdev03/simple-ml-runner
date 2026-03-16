from dataclasses import dataclass, field
from .base_callback import Callback
from engine.timer import Timer
from typing import Dict

@dataclass
class TimingCallback(Callback):
    """
    Manages job, train, eval, and epoch timers and records durations into metadata.
    """
    timers: Dict[str, Timer] = field(default_factory=lambda: {
        "job": Timer(),
        "train": Timer(),
        "eval": Timer(),
        "epoch": Timer()
    })

    def on_train_start(self, engine):
        self.timers["job"].start()
        self.timers["train"].start()

    def on_epoch_start(self, engine):
        # Fresh timer for each epoch as requested
        self.timers["epoch"] = Timer()
        self.timers["epoch"].start()

    def on_epoch_end(self, engine):
        self.timers["epoch"].pause()
        durations = {
            "epoch_duration": self.timers["epoch"].elapsed,
            "train_time_total": self.timers["train"].elapsed,
            "job_time_total": self.timers["job"].elapsed
        }
        engine.metadata.update("timing", durations)

    def on_train_end(self, engine):
        self.timers["train"].pause()
        self.timers["job"].pause()

    def on_eval_start(self, engine):
        # job timer may or may not be running depending on if eval is standalone
        self.timers["job"].start()
        self.timers["eval"].start()
        # Separate timer for evaluation phase epoch
        self.timers["epoch_eval"] = Timer()
        self.timers["epoch_eval"].start()

    def on_eval_end(self, engine):
        self.timers["epoch_eval"].pause()
        self.timers["eval"].pause()
        self.timers["job"].pause()

        durations = engine.metadata.get("timing", {})
        if not isinstance(durations, dict): durations = {}

        # If we were in a training epoch, we can combine or report separately
        # For now, let's update with eval duration and total job time
        durations["eval_epoch_duration"] = self.timers["epoch_eval"].elapsed
        durations["eval_time_total"] = self.timers["eval"].elapsed
        durations["job_time_total"] = self.timers["job"].elapsed

        # Update epoch_duration to include eval if it was just training
        train_dur = durations.get("epoch_duration", 0)
        durations["epoch_duration"] = train_dur + self.timers["epoch_eval"].elapsed

        engine.metadata.update("timing", durations)
