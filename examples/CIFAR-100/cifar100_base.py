from torchvision import datasets, transforms

CONFIG = "../shared/base_config.py"

_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

TRAIN_DATASET = datasets.CIFAR100(
    './data', train=True, download=True, transform=_transform
)

TEST_DATASET = datasets.CIFAR100(
    './data', train=False, download=True, transform=_transform
)

CHECK_MODEL_DIR = "checkpoints/cifar100/"
CHECK_MODEL_NAME = "cifar100_epoch_$epoch"

FINAL_OUTPUT_PATH = "models/cifar100_final.pt"

SAVE_TESTS = "results/cifar100_results.csv"

PROFILE_NAME = "cifar100_profile.csv"

BATCH_SIZE = 256
EPOCHS = 100
