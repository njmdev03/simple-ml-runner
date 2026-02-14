from examples.shared.ViT import VisionTransformer

CONFIG = "cifar100_base.py"

# CIFAR-100: 32x32 RGB, 100 classes
MODEL = VisionTransformer(
    img_size=32,
    patch_size=4,
    in_channels=3,
    num_classes=100,
    embed_dim=128,
    depth=6,
    num_heads=4,
    mlp_ratio=4.0,
    dropout=0.1
)


FINAL_OUTPUT_PATH = "models/cifar100_vit_final.pt"
CHECK_MODEL_DIR = "checkpoints/cifar100-vit/"
SAVE_TESTS = "results/cifar100_vit_results.csv"
PROFILE_OUTPUT = "profiles/cifar100_vit_profile.csv"
VIS_OUTPUT_DIR = "vis/vit/"

BATCH_SIZE = 256
EPOCHS = 10
