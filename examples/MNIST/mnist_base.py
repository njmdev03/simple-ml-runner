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

TESTING_BATCH_SIZE = 10000

BATCH_SIZE = 128
EPOCHS = 10
