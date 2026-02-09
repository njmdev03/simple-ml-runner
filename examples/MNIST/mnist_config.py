import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
# MNIST Configuration Demo

DEVICES = ["cuda", "cpu"]

_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

TRAIN_DATASET = datasets.MNIST(
    './data', train=True, download=True, transform=_transform
)

TEST_DATASET = datasets.MNIST(
    './data', train=False, download=True, transform=_transform
)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
MODEL = Net()

# Training Options
# RESUME = True
BATCH_SIZE = 8000
EPOCHS = 10
LEARNING_RATE = 0.001
# OPTIMIZER = "Adam"
OPTIMIZER = optim.Adam
# TRAIN_CRITERION = "NLLLoss"
TRAIN_CRITERION = nn.NLLLoss()
EARLY_HALT_CONDITION = "Accuracy"
EARLY_HALT_THRESHOLD = 0.95

# Checkpoint Options
CHECK_RATE = 1
CHECK_MODEL_DIR = "checkpoints/mnist/"
CHECK_MODEL_NAME = "mnist_epoch_$epoch"

# Output Options
FINAL_OUTPUT_PATH = "models/mnist_final.pt"

# Test Options
TEST_CHECKPOINTS = True
# TEST_WHILE_TRAINING = True
TEST_ON_TRAINING_DATA = True
TESTING_CRITERION = [nn.NLLLoss(), nn.CrossEntropyLoss()]
SAVE_TESTS = "results/mnist_results.csv"

# Profiling
PROFILE = True
PROFILE_DIR = "profiles/"
PROFILE_NAME = "mnist_profile.csv"
