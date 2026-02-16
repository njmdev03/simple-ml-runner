from examples.shared.ViT import VisionTransformer

CONFIG = "mnist_base.py"

# MNIST: 28x28 grayscale, 10 classes
MODEL = VisionTransformer(
    img_size=28,
    patch_size=4,
    in_channels=1,
    num_classes=10,
    embed_dim=64,
    depth=4,
    num_heads=4,
    mlp_ratio=4.0,
    dropout=0.1
)

FINAL_OUTPUT_PATH = "models/mnist_vit_final.pt"
CHECK_MODEL_DIR = "checkpoints/mnist-vit/"
SAVE_TESTS = "results/mnist_vit_results.csv"
PROFILE_OUTPUT = "profiles/mnist_vit_profile.csv"
VIS_OUTPUT_DIR = "vis/vit/"
