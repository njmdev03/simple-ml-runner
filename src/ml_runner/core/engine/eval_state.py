from dataclasses import dataclass, field

from ml_runner.core.engine.state import State


@dataclass
class EvalState(State):
    # Dictionary of metrics values by metric name
    metrics: dict = field(default_factory=dict)
    # Running total of metric values
    total_metrics: dict = field(default_factory=dict)

    def _safe_sum(self, dict_a, dict_b, key):
        a = dict_a.get(key)
        b = dict_b.get(key)

        if a is not None and b is not None:
            return a + b
        elif a is None:
            return b
        else:
            return a

    def get_averaged_metrics(self) -> dict:
        """
        Returns a dictionary of averaged metrics.
        """
        if self.batch == 0:
            return {}

        results = {}
        for k, v in self.total_metrics.items():
            results[k] = v / self.batch
        return results

    def sum_metrics(self, metrics: dict):
        for key in metrics.keys():
            self.total_metrics[key] = self._safe_sum(self.total_metrics, metrics, key)
