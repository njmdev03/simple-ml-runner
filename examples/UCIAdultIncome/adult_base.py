import torch
from adult_dataset import AdultDataset
from utils.tabular_normalize import TabularNormalize

CONFIG = "../shared/base_config.py"

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

# Test dataset needs encoders from training dataset
TEST_DATASET = AdultDataset(
    './data', train=False, download=True, transform=_transform,
    encoders=TRAIN_DATASET.encoders,
    target_encoder=TRAIN_DATASET.target_encoder
)

# Load fewer items in training to allow for gradient propagation
BATCH_SIZE = 128
EPOCHS = 10

# Load the whole dataset when testing for speed
TESTING_BATCH_SIZE = 20000
