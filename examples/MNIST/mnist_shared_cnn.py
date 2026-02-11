from examples.shared.CNN import CNN

CONFIG = "mnist_base.py"

FINAL_OUTPUT_PATH = "models/mnist_cnn_final.pt"

CHECK_MODEL_DIR = "checkpoints/mnist-cnn/"

SAVE_TESTS = "results/mnist_cnn_results.csv"

PROFILE_NAME = "mnist_cnn_profile.csv"

MODEL = CNN(28 * 28 ,10)
