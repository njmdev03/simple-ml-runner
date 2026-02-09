import torch
import torch.nn as nn
import torch.optim as optim

from adult_dataset import AdultDataset

from utils.tabular_normalize import TabularNormalize

# UCI Adult Income Configuration

DEVICES = ["cuda", "cpu"]

# Mean and Standard Deviation calculated using stats mode, then hard coded here.
_transform = TabularNormalize(
    mean=torch.tensor([3.8438e+01, 2.1993e+00, 1.8979e+05, 1.0334e+01, 1.0121e+01, 2.5801e+00,
        5.9599e+00, 1.4183e+00, 3.6786e+00, 6.7568e-01, 1.0920e+03, 8.8372e+01,
        4.0931e+01, 3.6383e+01]),
    std=torch.tensor([1.3134e+01, 9.5391e-01, 1.0565e+05, 3.8122e+00, 2.5500e+00, 1.4980e+00,
        4.0295e+00, 1.6013e+00, 8.3470e-01, 4.6812e-01, 7.4062e+03, 4.0429e+02,
        1.1980e+01, 6.1053e+00])
)

TRAIN_DATASET = AdultDataset(
    './data', train=True, download=True, transform=_transform
)
TEST_DATASET = AdultDataset(
    './data', train=False, download=True, transform=_transform,
    encoders=TRAIN_DATASET.encoders,
    target_encoder=TRAIN_DATASET.target_encoder
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        return self.out(x)

MODEL = Net()

# Training Options
BATCH_SIZE = 512
EPOCHS = 30
LEARNING_RATE = 0.001
OPTIMIZER = optim.Adam
TRAIN_CRITERION = nn.CrossEntropyLoss()
# EARLY_HALT_CONDITION = "Accuracy"
# EARLY_HALT_THRESHOLD = 0.86

# Checkpoint Options
CHECK_RATE = 1
CHECK_MODEL_DIR = "checkpoints/adult/"
CHECK_MODEL_NAME = "adult_epoch_$epoch"

# Output
FINAL_OUTPUT_PATH = "models/adult_final.pt"

# Testing
TEST_CHECKPOINTS = True
TEST_ON_TRAINING_DATA = False
TESTING_CRITERION = [nn.CrossEntropyLoss()]
SAVE_TESTS = "results/adult_results.csv"

# Profiling
PROFILE = True
PROFILE_DIR = "profiles/"
PROFILE_NAME = "adult_profile.csv"
