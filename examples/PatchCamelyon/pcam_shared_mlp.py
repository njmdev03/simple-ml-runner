from examples.shared.MLP import MLP

CONFIG = "pcam_base.py"

# PCAM images are 96x96x3 = 27648 input features
# 2 output classes (tumor / no tumor)
MODEL = MLP(27648, 2)

FINAL_OUTPUT_PATH = "models/pcam_mlp_final.pt"
CHECK_MODEL_DIR = "checkpoints/pcam-mlp/"
SAVE_TESTS = "results/pcam_mlp_results.csv"
PROFILE_NAME = "pcam_mlp_profile.csv"
