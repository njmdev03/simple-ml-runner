from MLP import MLP

CONFIG = "cifar100_base.py"

# CIFAR-100 images are 32x32x3 = 3072 input features
# 100 output classes
MODEL = MLP(3072, 100, [3072, 2048, 2048, 1024, 512, 256])

FINAL_OUTPUT_PATH = "models/cifar100_mlp_final.pt"
CHECK_MODEL_DIR = "checkpoints/cifar100-mlp/"
SAVE_TESTS = "results/cifar100_mlp_results.csv"
PROFILE_OUTPUT = "profiles/cifar100_mlp_profile.csv"
VIS_OUTPUT_DIR = "vis/mlp/"

BATCH_SIZE = 256
EPOCHS = 100
