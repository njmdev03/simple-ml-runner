import time
from typing import Dict, List, Any

class Profiler:
    def __init__(self):
        # self.start_time = None
        # self.end_time = None
        self.durations = {}
        # self.epoch_times = []
        self.started = {}

    def start(self, name: str = "total"):
        self.started[name] = time.perf_counter()

    def is_active(self, name: str) -> bool:
        return name in self.started and name not in self.durations

    def stop(self, name: str = "total"):
        start = self.started.get(name)
        if start:
            duration = time.perf_counter() - start
            self.durations[name] = duration
            # if name == "total":
            #     self.end_time = time.perf_counter()
            return duration
        return 0

    def get_duration(self, name: str = "total"):
        if self.durations[name]:
            return self.durations[name]
        elif self.started[name]:
            return self.started[name]
        else:
            return -1

    # def record_epoch(self, epoch: int, duration: float):
    #     self.epoch_times.append({
    #         "epoch": epoch,
    #         "duration": duration
    #     })

    def get_report(self) -> Dict[str, Any]:
        # report = {
        #     "durations": self.durations,
        #     "epochs": self.epoch_times
        # }

        # return report

        return self.durations

    # def get_flat_report(self) -> List[Dict[str, Any]]:
    #     flat = []
    #     # Main durations (Total, Training, Testing)
    #     for name, duration in self.durations.items():
    #         if not name.startswith("epoch_"):
    #             flat.append({"metric": f"{name}_duration", "value": duration})
    #     # Epochs
    #     for item in self.epoch_times:
    #         flat.append({"metric": f"epoch_{item['epoch']}_duration", "value": item['duration']})
    #     return flat
