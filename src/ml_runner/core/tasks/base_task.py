class BaseTask:
    def __init__(self, model, loss_fn, optimizer, train_loader, val_loader, device="cpu", metrics=None):
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metrics = metrics or []

    def training_step(self, batch):
        raise NotImplementedError

    def evaluation_step(self, batch):
        raise NotImplementedError

    def compute_metrics(self, outputs, targets):
        results = {}

        for metric in self.metrics:
            results[metric.__name__] = metric(outputs, targets)

        return results

    def load_checkpoint(self, path):
        import torch
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)
