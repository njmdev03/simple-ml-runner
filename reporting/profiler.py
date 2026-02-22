import time
from typing import Dict, Any
import re

class Profiler:
    def __init__(self, existing_durations: Dict[str, float] = None):
        self.durations = existing_durations or {}
        self.started = {}

    def _get_unique_name(self, name: str) -> str:
        if name not in self.durations and name not in self.started:
            return name

        # Find highest suffix like name_1, name_2
        pattern = re.compile(rf"^{re.escape(name)}_(\d+)$")
        max_idx = 0

        for k in list(self.durations.keys()) + list(self.started.keys()):
            match = pattern.match(k)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
            elif k == name:
                # The base name exists, so we at least need _1
                max_idx = max(max_idx, 0)

        return f"{name}_{max_idx + 1}"

    def start(self, name: str = "total", unique: bool = False):
        if unique:
            name = self._get_unique_name(name)
        self.started[name] = time.perf_counter()
        return name

    def resume(self, name: str = "total"):
        """Alias for start, used to clarify intent when continuing accumulated time."""
        return self.start(name, unique=False)

    def is_active(self, name: str) -> bool:
        return name in self.started

    def stop(self, name: str = "total"):
        start_time = self.started.get(name)
        if start_time is not None:
            duration = time.perf_counter() - start_time
            # Accumulate duration
            self.durations[name] = self.durations.get(name, 0.0) + duration
            del self.started[name]
            return self.durations[name]
        return self.durations.get(name, 0.0)

    def pause(self, name: str = "total"):
        """Alias for stop, used to clarify intent when pausing accumulated time."""
        return self.stop(name)

    def get_duration(self, name: str = "total"):
        total = self.durations.get(name, 0.0)
        if self.is_active(name):
            total += time.perf_counter() - self.started[name]
        return total

    def get_report(self) -> Dict[str, Any]:
        return self.durations
