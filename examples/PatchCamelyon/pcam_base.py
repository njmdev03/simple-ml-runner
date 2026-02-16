from torchvision import datasets, transforms

CONFIG = "../shared/base_config.py"

_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.7008, 0.5384, 0.6916),
        (0.2350, 0.2774, 0.2129)
    )
])

# PCAM has split argument instead of train
TRAIN_DATASET = datasets.PCAM(
    './data', split='train', download=True, transform=_transform
)

TEST_DATASET = datasets.PCAM(
    './data', split='test', download=True, transform=_transform
)

TESTING_BATCH_SIZE = 512
