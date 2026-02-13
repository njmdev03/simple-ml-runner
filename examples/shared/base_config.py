"""
Base configuration used by shared example configs
"""

import torch.optim as optim
import torch.nn as nn

DEVICES = ["cuda", "cpu"]

LEARNING_RATE = 0.003
OPTIMIZER = optim.Adam
TRAIN_CRITERION = nn.CrossEntropyLoss()

CHECK_RATE = 1

TEST_CHECKPOINTS = True
TEST_ON_TRAINING_DATA = True

PROFILE = True

# Select visualizations
VIS_TYPE = ["all"]
VIS_METRICS = ["all"]
VIS_DATASETS = ["all"]
NUM_SAMPLES = 10
SHOW = False
