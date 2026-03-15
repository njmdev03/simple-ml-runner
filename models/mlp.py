import torch.nn as nn
from registries import ModelRegistry


@ModelRegistry.register("MLP")
class MLP(nn.Module):
    def __init__(self, input_dim: int, layers: list, output_dim: int):
        """
        input_dim: number of features in input
        layers: list of hidden layer sizes, e.g. [128, 64]
        output_dim: number of classes
        """
        super().__init__()

        dims = [input_dim] + layers + [output_dim]
        modules = []

        for i in range(len(dims) - 1):
            modules.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                modules.append(nn.ReLU())

        self.net = nn.Sequential(*modules)

    def forward(self, x):
        return self.net(x)
