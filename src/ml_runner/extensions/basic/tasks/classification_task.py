import torch
from ml_runner.core.tasks.base_task import BaseTask
from ml_runner.core.registries import Task


@Task("classification")
class ClassificationTask(BaseTask):

    def training_step(self, batch):

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        outputs = self.model(x)

        loss = self.loss_fn(outputs, y)

        return loss, outputs, y

    def evaluation_step(self, batch):

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        with torch.no_grad():
            outputs = self.model(x)

        loss = self.loss_fn(outputs, y)

        return loss, outputs, y
