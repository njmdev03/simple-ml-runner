from examples.shared.CNN import CNN

CONFIG = "cifar100_base.py"

# CIFAR-100 images are 3 channels
# 100 output classes
MODEL = CNN(3, 100)

FINAL_OUTPUT_PATH = "models/cifar100_cnn_final.pt"
CHECK_MODEL_DIR = "checkpoints/cifar100-cnn/"
SAVE_TESTS = "results/cifar100_cnn_results.csv"
PROFILE_OUTPUT = "profiles/cifar100_cnn_profile.csv"
VIS_OUTPUT_DIR = "vis/cnn/"

BATCH_SIZE = 3000
EPOCHS = 10
