import time


class Timer:
    """
    A simple timer with pause/resume capability.
    """
    def __init__(self):
        self._start_time = 0
        self._elapsed = 0
        self._is_running = False

    def start(self):
        if not self._is_running:
            self._start_time = time.time()
            self._is_running = True

    def pause(self):
        if self._is_running:
            self._elapsed += time.time() - self._start_time
            self._is_running = False

    def reset(self):
        self._start_time = 0
        self._elapsed = 0
        self._is_running = False

    @property
    def elapsed(self):
        current_elapsed = self._elapsed
        if self._is_running:
            current_elapsed += time.time() - self._start_time
        return current_elapsed
