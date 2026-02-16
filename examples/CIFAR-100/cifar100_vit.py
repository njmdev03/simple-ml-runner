import torch.optim as optim
import torch.nn as nn
from examples.shared.ViT import VisionTransformer

CONFIG = "cifar100_base.py"

# CIFAR-100: 32x32 RGB, 100 classes
MODEL = VisionTransformer(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=100,
    embed_dim=256,
    depth=6,
    num_heads=8,
    mlp_ratio=4.0,
    dropout=0.3
)


FINAL_OUTPUT_PATH = "models/cifar100_vit_final.pt"
CHECK_MODEL_DIR = "checkpoints/cifar100-vit/"
SAVE_TESTS = "results/cifar100_vit_results.csv"
PROFILE_OUTPUT = "profiles/cifar100_vit_profile.csv"
VIS_OUTPUT_DIR = "vis/vit/"

BATCH_SIZE = 256
EPOCHS = 100

# ViT Overrides for CIFAR-100
LEARNING_RATE = 5e-4
# Use a lambda to pass weight_decay to AdamW
OPTIMIZER = lambda params, lr: optim.AdamW(params, lr=lr, weight_decay=0.1)
# Label smoothing helps reduce overfitting in 100-class problems
TRAIN_CRITERION = nn.CrossEntropyLoss(label_smoothing=0.1)
