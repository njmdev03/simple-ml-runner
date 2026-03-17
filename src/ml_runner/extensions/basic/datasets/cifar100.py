from torchvision.datasets import CIFAR100
from torchvision import transforms

from ml_runner.core.registries import Dataset


@Dataset("CIFAR100")
class CIFAR100Dataset(Dataset):
    def __init__(self, train=True, download=True, transform=None):
        self.dataset = CIFAR100(
            root="./data/cifar100",
            train=train,
            download=download,
            transform=transform or transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
