from torchvision.datasets import OxfordIIITPet
from torchvision import transforms

from ml_runner.core.registries import Dataset


@Dataset("OxfordPet")
class OxfordPetDataset(Dataset):
    def __init__(self, split="trainval", transform=None, target_transform=None, download=True):
        self.dataset = OxfordIIITPet(
            root="./data/oxford_pet",
            split=split,
            download=download,
            transform=transform or transforms.ToTensor(),
            target_transform=target_transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]