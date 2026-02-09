import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# PatchCamelyon Configuration

DEVICES = ["cuda", "cpu"]

_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.7000, 0.5000, 0.7000),
        (0.2000, 0.2000, 0.2000)
    )
])

TRAIN_DATASET = datasets.PCAM(
    './data', split='train', download=True, transform=_transform
)

TEST_DATASET = datasets.PCAM(
    './data', split='test', download=True, transform=_transform
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 12 * 12, 256)
        self.fc2 = nn.Linear(256, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

MODEL = Net()

# Training
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.0005
OPTIMIZER = optim.Adam
TRAIN_CRITERION = nn.CrossEntropyLoss()
EARLY_HALT_CONDITION = "Accuracy"
EARLY_HALT_THRESHOLD = 0.88

# Checkpoints
CHECK_RATE = 1
CHECK_MODEL_DIR = "checkpoints/pcam/"
CHECK_MODEL_NAME = "pcam_epoch_$epoch"

FINAL_OUTPUT_PATH = "models/pcam_final.pt"

# Testing
TEST_CHECKPOINTS = True
TEST_ON_TRAINING_DATA = False
TESTING_CRITERION = [nn.CrossEntropyLoss()]
SAVE_TESTS = "results/pcam_results.csv"

# Profiling
PROFILE = True
PROFILE_DIR = "profiles/"
PROFILE_NAME = "pcam_profile.csv"
