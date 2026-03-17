from torchvision.datasets import MNIST
from torchvision import transforms

from ml_runner.core.registries import Dataset


@Dataset("MNIST")
class MNISTDataset(Dataset):
    def __init__(self, train=True, download=True, root="./data/mnist", transform=None):
        self.dataset = MNIST(
            root=root,
            train=train,
            download=download,
            transform=transform or transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]