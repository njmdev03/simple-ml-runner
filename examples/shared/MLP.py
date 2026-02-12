import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim1: int = 512,
        hidden_dim2: int = 256,
        dropout_p: float = 0.5,
        use_batchnorm: bool = True,
    ):
        """
        Fully connected feedforward network with:
        - Input layer
        - 2 hidden layers (ReLU)
        - Output layer
        - Dropout
        - Optional BatchNorm
        """

        super().__init__()

        self.flatten = nn.Flatten()

        # Layer 1
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.bn1 = nn.BatchNorm1d(hidden_dim1) if use_batchnorm else nn.Identity()

        # Layer 2
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.bn2 = nn.BatchNorm1d(hidden_dim2) if use_batchnorm else nn.Identity()

        # Output layer
        self.fc3 = nn.Linear(hidden_dim2, num_classes)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        # Flatten in case input is image
        x = self.flatten(x)

        # Hidden layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Hidden layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Output layer (no softmax — use CrossEntropyLoss)
        x = self.fc3(x)

        return x
