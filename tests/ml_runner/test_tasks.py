import torch
import ml_runner.extensions  # ensure basic extensions (and tasks) are registered
from ml_runner.core.registries import TaskRegistry

def test_classification_task_compute_metrics():
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def mock_metric(outputs, targets):
        return 1.0

    TaskClass = TaskRegistry.get("classification")

    task = TaskClass(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_loader=[],
        val_loader=[],
        metrics=[mock_metric]
    )

    outputs = torch.randn(5, 2)
    targets = torch.randint(0, 2, (5,))

    results = task.compute_metrics(outputs, targets)
    assert results["mock_metric"] == 1.0

def test_classification_task_steps():
    model = torch.nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    TaskClass = TaskRegistry.get("classification")
    task = TaskClass(
        model=model,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        train_loader=[],
        val_loader=[]
    )

    batch = (torch.randn(5, 10), torch.randint(0, 2, (5,)))

    # Train step
    loss, outputs, targets = task.training_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert outputs.shape == (5, 2)

    # Eval step
    loss, outputs, targets = task.evaluation_step(batch)
    assert isinstance(loss, torch.Tensor)
    assert outputs.shape == (5, 2)
