from abc import ABC, abstractmethod

class BaseMetric(ABC):
    @abstractmethod
    def update(self, output, target):
        pass

    @abstractmethod
    def compute(self) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass

class Accuracy(BaseMetric):
    def __init__(self):
        self.correct = 0
        self.total = 0

    def update(self, output, target):
        pred = output.argmax(dim=1, keepdim=True)
        self.correct += pred.eq(target.view_as(pred)).sum().item()
        self.total += target.size(0)

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0

class Precision(BaseMetric):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.tp = 0
        self.fp = 0

    def update(self, output, target):
        pred = output.argmax(dim=1)
        if self.num_classes is None:
            self.num_classes = output.size(1)

        for c in range(self.num_classes):
            self.tp += ((pred == c) & (target == c)).sum().item()
            self.fp += ((pred == c) & (target != c)).sum().item()

    def compute(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    def reset(self):
        self.tp = 0
        self.fp = 0

class Recall(BaseMetric):
    def __init__(self, num_classes=None):
        self.num_classes = num_classes
        self.tp = 0
        self.fn = 0

    def update(self, output, target):
        pred = output.argmax(dim=1)
        if self.num_classes is None:
            self.num_classes = output.size(1)

        for c in range(self.num_classes):
            self.tp += ((pred == c) & (target == c)).sum().item()
            self.fn += ((pred != c) & (target == c)).sum().item()

    def compute(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    def reset(self):
        self.tp = 0
        self.fn = 0

class F1Score(BaseMetric):
    def __init__(self):
        self.precision = Precision()
        self.recall = Recall()

    def update(self, output, target):
        self.precision.update(output, target)
        self.recall.update(output, target)

    def compute(self) -> float:
        p = self.precision.compute()
        r = self.recall.compute()
        denom = p + r
        return 2 * (p * r) / denom if denom > 0 else 0.0

    def reset(self):
        self.precision.reset()
        self.recall.reset()

class MeanIoU(BaseMetric):
    def __init__(self, num_classes=None, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.intersection = 0
        self.union = 0

    def update(self, output, target):
        pred = output.argmax(dim=1)
        if self.num_classes is None:
            self.num_classes = output.size(1)

        mask = (target != self.ignore_index)
        pred = pred[mask]
        target = target[mask]

        for c in range(self.num_classes):
            inter = ((pred == c) & (target == c)).sum().item()
            uni = ((pred == c) | (target == c)).sum().item()
            self.intersection += inter
            self.union += uni

    def compute(self) -> float:
        return self.intersection / self.union if self.union > 0 else 0.0

    def reset(self):
        self.intersection = 0
        self.union = 0

class PixelAccuracy(BaseMetric):
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        self.correct = 0
        self.total = 0

    def update(self, output, target):
        pred = output.argmax(dim=1)
        mask = (target != self.ignore_index)
        self.correct += (pred[mask] == target[mask]).sum().item()
        self.total += mask.sum().item()

    def compute(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    def reset(self):
        self.correct = 0
        self.total = 0
