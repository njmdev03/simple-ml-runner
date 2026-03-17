from dataclasses import dataclass, field
from .base_callback import Callback
import torch
from pathlib import Path
from ml_runner.core.log_utils import logger

@dataclass
class TorchProfilerCallback(Callback):
    """
    Integrates PyTorch Profiler for performance analysis.
    """
    output_dir: str = "profiles/"
    wait: int = 1
    warmup: int = 1
    active: int = 3
    repeat: int = 1

    def __post_init__(self):
        self.profiler = None

    def on_train_start(self, engine):
        self.profiler = torch.profiler.profile(
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=True,
            with_stack=True
        )
        self.profiler.start()
        logger.info(f"Torch Profiler started, output to {self.output_dir}")

    def on_batch_end(self, engine):
        if self.profiler:
            self.profiler.step()

    def on_train_end(self, engine):
        if self.profiler:
            self.profiler.stop()
            logger.info("Torch Profiler stopped")
