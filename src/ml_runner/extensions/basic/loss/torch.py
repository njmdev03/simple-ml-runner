import torch.nn as nn
from ml_runner.core.registries import Loss


Loss("cross_entropy")(nn.CrossEntropyLoss)
Loss("mse")(nn.MSELoss)
Loss("nll")(nn.NLLLoss)
