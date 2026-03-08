import cProfile
from logging import logger
from dataclasses import dataclass
from .base_callback import Callback

@dataclass
class FlameProfiler(Callback):

    def on_train_start(self, engine):

        self.profiler = cProfile.Profile()
        self.profiler.enable()

    def on_train_end(self, engine):

        self.profiler.disable()
        self.profiler.dump_stats("profile.prof")

        logger.info(f"Profile saved to profile.prof")