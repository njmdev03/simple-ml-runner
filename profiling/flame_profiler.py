import cProfile
from logging import logger


class FlameProfiler:

    def __init__(self, output="profile.prof"):
        self.profiler = cProfile.Profile()
        self.output = output

    def start(self):
        self.profiler.enable()

    def stop(self):

        self.profiler.disable()

        self.profiler.dump_stats(self.output)

        logger.info(f"Profile written to {self.output}")