import torch
import torch.nn as nn
import torch.nn.functional as F



class MLP(nn.Module):
    """
    A configurable Multi-Layer Perceptron (MLP).

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): Sizes of hidden layers.
        output_dim (int): Number of output features/classes.
        activation (nn.Module): Activation function class.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation())

            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
