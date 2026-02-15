from torchvision import datasets, transforms

CONFIG = "../shared/base_config.py"

# Set up the transforms for training and testing. We use data augmentation for training, but not for testing.
_train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

_test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5071, 0.4867, 0.4408),
        (0.2675, 0.2565, 0.2761)
    )
])

TRAIN_DATASET = datasets.CIFAR100(
    './data', train=True, download=True, transform=_train_transform
)

TEST_DATASET = datasets.CIFAR100(
    './data', train=False, download=True, transform=_test_transform
)
