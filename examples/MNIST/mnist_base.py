from torchvision import datasets, transforms

CONFIG = "../shared/base_config.py"

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

TRAIN_DATASET = datasets.MNIST(
    './data', train=True, download=True, transform=_transform
)

TEST_DATASET = datasets.MNIST(
    './data', train=False, download=True, transform=_transform
)

CHECK_MODEL_DIR = "checkpoints/mnist/"
CHECK_MODEL_NAME = "mnist_epoch_$epoch"

FINAL_OUTPUT_PATH = "models/mnist_final.pt"

SAVE_TESTS = "results/mnist_results.csv"

PROFILE_NAME = "mnist_profile.csv"

BATCH_SIZE = 8000
EPOCHS = 10
