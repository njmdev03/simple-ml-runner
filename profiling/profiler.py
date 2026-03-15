import time
from log_utils import logger


class Profiler:

    def __init__(self):
        self.times = {}
        self.starts = {}

    def start(self, name):
        self.starts[name] = time.time()

    def stop(self, name):

        elapsed = time.time() - self.starts[name]

        self.times[name] = self.times.get(name, 0) + elapsed

    def summary(self):

        logger.info("\n=== Profiling Summary ===")

        for name, t in self.times.items():
            logger.info(f"{name}: {t:.3f}s")