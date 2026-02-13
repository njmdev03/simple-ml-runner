from examples.shared.CNN import CNN

CONFIG = "mnist_base.py"

FINAL_OUTPUT_PATH = "models/mnist_cnn_final.pt"

CHECK_MODEL_DIR = "checkpoints/mnist-cnn/"

SAVE_TESTS = "results/mnist_cnn_results.csv"

PROFILE_OUTPUT = "profiles/mnist_cnn_profile.csv"

VIS_OUTPUT_DIR = "vis/cnn/"

MODEL = CNN(1 ,10)

EPOCHS = 10
BATCH_SIZE = 8000
