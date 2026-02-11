from torchvision import datasets, transforms

from examples.shared.MLP import MLP

CONFIG = "../shared/base_config.py"

_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.7000, 0.5000, 0.7000),
        (0.2000, 0.2000, 0.2000)
    )
])

TRAIN_DATASET = datasets.PCAM(
    './data', split='train', download=True, transform=_transform
)

TEST_DATASET = datasets.PCAM(
    './data', split='test', download=True, transform=_transform
)

MODEL = MLP(28 * 28 ,10)

CHECK_MODEL_DIR = "checkpoints/mnist/"
CHECK_MODEL_NAME = "mnist_epoch_$epoch"

FINAL_OUTPUT_PATH = "models/mnist_final.pt"

SAVE_TESTS = "results/mnist_results.csv"

PROFILE_NAME = "mnist_profile.csv"
