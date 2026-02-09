import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# CIFAR-100 Configuration

DEVICES = ["cuda", "cpu"]

_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

TRAIN_DATASET = datasets.CIFAR100(
    './data', train=True, download=True, transform=_transform
)

TEST_DATASET = datasets.CIFAR100(
    './data', train=False, download=True, transform=_transform
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

MODEL = Net()

# Training Options
BATCH_SIZE = 256
EPOCHS = 100
LEARNING_RATE = 0.001
OPTIMIZER = optim.Adam
TRAIN_CRITERION = nn.CrossEntropyLoss()
EARLY_HALT_CONDITION = "Accuracy"
EARLY_HALT_THRESHOLD = 0.60  # realistic baseline

# Checkpoints
CHECK_RATE = 5
CHECK_MODEL_DIR = "checkpoints/cifar100/"
CHECK_MODEL_NAME = "cifar100_epoch_$epoch"

FINAL_OUTPUT_PATH = "models/cifar100_final.pt"

# Testing
TEST_CHECKPOINTS = True
TEST_ON_TRAINING_DATA = False
TESTING_CRITERION = [nn.CrossEntropyLoss()]
SAVE_TESTS = "results/cifar100_results.csv"

# Profiling
PROFILE = True
PROFILE_DIR = "profiles/"
PROFILE_NAME = "cifar100_profile.csv"
