import torch
import torch.nn as nn
from examples.shared.CNN import CNN

CONFIG = "adult_base.py"

class TabularToImage(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x shape: (batch_size, 14)

        # 1. Pad to 16 features
        batch_size = x.size(0)
        padding = torch.zeros(batch_size, 2, device=x.device)
        x_padded = torch.cat([x, padding], dim=1) # (batch_size, 16)

        # 2. Reshape to (batch_size, 1, 4, 4)
        x_image = x_padded.view(batch_size, 1, 4, 4)

        return x_image

# Wrap the CNN with the adapter
# Input: 14 features -> Padded to 16 -> Reshaped to 1x4x4
# CNN takes 1 channel input, 2 classes output
MODEL = nn.Sequential(
    TabularToImage(),
    CNN(1, 2)
)

FINAL_OUTPUT_PATH = "models/adult_cnn_final.pt"
CHECK_MODEL_DIR = "checkpoints/adult-cnn/"
SAVE_TESTS = "results/adult_cnn_results.csv"
PROFILE_OUTPUT = "profiles/adult_cnn_profile.csv"
VIS_OUTPUT_DIR = "vis/cnn/"
