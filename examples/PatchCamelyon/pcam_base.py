from torchvision import datasets, transforms

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

# PCAM has split argument instead of train
TRAIN_DATASET = datasets.PCAM(
    './data', split='train', download=True, transform=_transform
)

TEST_DATASET = datasets.PCAM(
    './data', split='test', download=True, transform=_transform
)

CHECK_MODEL_DIR = "checkpoints/pcam/"
CHECK_MODEL_NAME = "pcam_epoch_$epoch"

FINAL_OUTPUT_PATH = "models/pcam_final.pt"

SAVE_TESTS = "results/pcam_results.csv"

PROFILE_NAME = "pcam_profile.csv"

BATCH_SIZE = 128
EPOCHS = 20
