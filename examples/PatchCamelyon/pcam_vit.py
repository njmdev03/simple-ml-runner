import torch.optim as optim
from examples.shared.ViT import VisionTransformer

CONFIG = "pcam_base.py"

# PatchCamelyon: 96x96 RGB, 2 classes
MODEL = VisionTransformer(
    img_size=96,
    patch_size=6,
    in_channels=3,
    num_classes=2,
    embed_dim=384,
    depth=12,
    num_heads=12,
    mlp_ratio=4.0,
    dropout=0.1
)


FINAL_OUTPUT_PATH = "models/pcam_vit_final.pt"
CHECK_MODEL_DIR = "checkpoints/pcam-vit/"
SAVE_TESTS = "results/pcam_vit_results.csv"
PROFILE_OUTPUT = "profiles/pcam_vit_profile.csv"
VIS_OUTPUT_DIR = "vis/vit/"

BATCH_SIZE = 256
EPOCHS = 10

# Custom learning required for
LEARNING_RATE = 8e-4  # Start even lower for stability
OPTIMIZER = lambda params, lr: optim.AdamW(params, lr=lr, weight_decay=0.01)
