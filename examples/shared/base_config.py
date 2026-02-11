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
SAVE_METADATA = True

TEST_CHECKPOINTS = True
TEST_ON_TRAINING_DATA = True

PROFILE = True
PROFILE_DIR = "profiles/"
