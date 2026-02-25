"""
Custom step functions for Object Detection with Faster R-CNN.

These are passed into the Config as TRAIN_STEP_FN and EVAL_STEP_FN.
The engine calls these instead of its standard forward-pass logic,
allowing detection models (which have their own internal loss and
use list-of-dict I/O) to work without any engine modifications.

Signature contracts:
    TRAIN_STEP_FN(model, batch, device) -> (loss: Tensor, output: None)
    EVAL_STEP_FN(model, batch, device)  -> (preds: list[dict], targets: list[dict])
"""

import torch


def detection_train_step(model, batch, device):
    """Training step for Faster R-CNN and compatible detection models.

    Faster R-CNN computes its own internal loss during the forward pass
    when targets are provided. We sum the loss_dict values into a single
    scalar for backpropagation.

    Returns:
        loss  (Tensor): Scalar loss for backward().
        output (None):  No prediction output during training; metrics
                        are evaluation-only for detection tasks.
    """
    images, targets = batch

    # Move to device
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Forward pass: model returns a dict of losses during training
    loss_dict = model(images, targets)
    loss = sum(l for l in loss_dict.values())

    return loss, None, targets  # Return targets so engine can update informational metrics


def detection_eval_step(model, batch, device):
    """Evaluation step for Faster R-CNN and compatible detection models.

    In eval mode, Faster R-CNN ignores the targets and returns a list
    of prediction dicts per image. We return both the predictions and
    the ground-truth targets in torchmetrics-compatible format.

    Returns:
        preds   (list[dict]): Each dict has 'boxes', 'scores', 'labels'.
        targets (list[dict]): Each dict has 'boxes', 'labels'.
    """
    images, targets = batch

    # Move to device
    images = [img.to(device) for img in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

    # Forward pass: model returns predictions in eval mode
    preds = model(images)   # list of dicts

    return preds, targets, None
