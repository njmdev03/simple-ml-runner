import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
# MNIST Configuration Demo

DEVICES = ["cuda", "cpu"]

class TinyRegressionDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples=32):
        torch.manual_seed(0)
        self.x = torch.linspace(-1, 1, n_samples).unsqueeze(1)
        self.y = 2.0 * self.x + 1.0   # <-- (N, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TinyClassificationDataset(Dataset):
    def __init__(self):
        self.x = torch.tensor(
            [[0.0], [1.0], [2.0], [3.0]],
            dtype=torch.float32
        )
        # IMPORTANT: 1D tensor, dtype long
        self.y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

TRAIN_DATASET = TinyRegressionDataset()
TEST_DATASET = TinyRegressionDataset()

class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

MODEL = TinyModel()

# Training Options
# RESUME = True
BATCH_SIZE = 100
EPOCHS = 2
LEARNING_RATE = 0.01
# OPTIMIZER = "SGD"
OPTIMIZER = optim.SGD
# TRAIN_CRITERION = "MSELoss"
TRAIN_CRITERION = nn.MSELoss()
EARLY_HALT_CONDITION = "Accuracy"
EARLY_HALT_THRESHOLD = 0.95

# Checkpoint Options
CHECK_RATE = 1
CHECK_MODEL_DIR = "checkpoints/"
CHECK_MODEL_NAME = "epoch_$epoch"

# Output Options
FINAL_OUTPUT_PATH = "./models/final.pt"

# Test Options
TEST_CHECKPOINTS = True
SAVE_METADATA = True
# TEST_WHILE_TRAINING = True
TEST_ON_TRAINING_DATA = True
TESTING_CRITERION = [nn.MSELoss(), nn.CrossEntropyLoss()]
SAVE_TESTS = "./results/results.csv"

# Profiling
PROFILE = True
PROFILE_OUTPUT = "./profiles/profile.csv"
