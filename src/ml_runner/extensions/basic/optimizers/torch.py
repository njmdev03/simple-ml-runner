import torch.optim as optim
from ml_runner.core.registries import Optimizer


Optimizer("adam")(optim.Adam)
Optimizer("sgd")(optim.SGD)
Optimizer("adamw")(optim.AdamW)
