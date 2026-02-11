from examples.shared.MLP import MLP

CONFIG = "mnist_base.py"

FINAL_OUTPUT_PATH = "models/mnist_mlp_final.pt"

CHECK_MODEL_DIR = "checkpoints/mnist-mlp/"

SAVE_TESTS = "results/mnist_mlp_results.csv"

PROFILE_NAME = "mnist_mlp_profile.csv"

MODEL = MLP(28 * 28 ,10)
