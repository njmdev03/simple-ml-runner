import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        layer_dims: list[int],      # list of hidden sizes
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        # build a stack of (Linear → [BatchNorm] → ReLU → Dropout)
        self.hidden = nn.ModuleList()
        prev = input_dim
        for dim in layer_dims:
            self.hidden.append(nn.Linear(prev, dim))
            if use_batchnorm:
                self.hidden.append(nn.BatchNorm1d(dim))
            self.hidden.append(nn.ReLU())
            self.hidden.append(nn.Dropout(dropout_p))
            prev = dim

        self.output = nn.Linear(prev, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        for layer in self.hidden:      # walk the ModuleList
            x = layer(x)
        x = self.output(x)
        return x
