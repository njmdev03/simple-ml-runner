import torch.optim as optim
from ml_runner.core.registries import Scheduler


Scheduler("step_lr")(optim.lr_scheduler.StepLR)
Scheduler("cosine_annealing")(optim.lr_scheduler.CosineAnnealingLR)
