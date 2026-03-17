from ml_runner.core.registries import Metric


@Metric("accuracy")
def accuracy(outputs, targets):
    preds = outputs.argmax(dim=1)
    return (preds == targets).float().mean().item()
