import torch
from ml_runner.core.registries import Metric

@Metric("precision")
def precision(outputs, targets):
    """
    Simple macro-averaged precision.
    """
    preds = outputs.argmax(dim=1)
    num_classes = outputs.size(1)
    
    # Ensure targets is 1D
    if targets.dim() > 1:
        targets = targets.view(-1)
        
    precisions = []
    for c in range(num_classes):
        true_positives = ((preds == c) & (targets == c)).sum().item()
        false_positives = ((preds == c) & (targets != c)).sum().item()
        if (true_positives + false_positives) > 0:
            precisions.append(true_positives / (true_positives + false_positives))
        # If no instances predicted, we don't contribute to macro average or count it as 0?
        # Macro-average usually counts it as 0.0 for that class or skips it.
    
    return sum(precisions) / len(precisions) if precisions else 0.0
