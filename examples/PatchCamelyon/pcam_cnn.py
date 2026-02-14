from examples.shared.CNN import CNN

CONFIG = "pcam_base.py"

# PCAM images are 3 channels
# 2 output classes
MODEL = CNN(3, 2)

FINAL_OUTPUT_PATH = "models/pcam_cnn_final.pt"
CHECK_MODEL_DIR = "checkpoints/pcam-cnn/"
SAVE_TESTS = "results/pcam_cnn_results.csv"
PROFILE_OUTPUT = "profiles/pcam_cnn_profile.csv"
VIS_OUTPUT_DIR = "vis/cnn/"

BATCH_SIZE = 600
EPOCHS = 3
