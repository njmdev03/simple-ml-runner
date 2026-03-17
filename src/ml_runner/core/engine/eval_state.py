from .state import State
from dataclasses import dataclass, field

@dataclass
class EvalState(State):
    # Dictionary of metrics values by metric name
    metrics: dict = field(default_factory=dict)
    # Running total of metric values
    total_metrics: dict = field(default_factory=dict)

    def _safe_sum(self, dict_a, dict_b, key):
        a = dict_a.get(key)
        b = dict_b.get(key)

        if a and b:
            return a + b
        elif (not a) and (not b):
            return None
        else:
            return a if not b else b

    def sum_metrics(self, metrics: dict):
        for key in metrics.keys():
            self.total_metrics[key] = self._safe_sum(self.total_metrics, metrics, key)