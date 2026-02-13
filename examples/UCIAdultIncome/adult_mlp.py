from examples.shared.MLP import MLP

CONFIG = "adult_base.py"

# Adult dataset has 14 features
# 2 output classes (>50k, <=50k)
MODEL = MLP(14, 2)

FINAL_OUTPUT_PATH = "models/adult_mlp_final.pt"
CHECK_MODEL_DIR = "checkpoints/adult-mlp/"
SAVE_TESTS = "results/adult_mlp_results.csv"
PROFILE_OUTPUT = "profiles/adult_mlp_profile.csv"
VIS_OUTPUT_DIR = "vis/mlp/"
