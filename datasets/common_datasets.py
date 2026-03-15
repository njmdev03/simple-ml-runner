# datasets/common_datasets.py
import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CIFAR100, OxfordIIITPet, PennFudanPed
from torchvision import transforms
from torch.utils.data import random_split
from registries import DatasetRegistry
import pandas as pd
from sklearn.model_selection import train_test_split

# -------------------------
# MNIST
# -------------------------
@DatasetRegistry.register("MNIST")
class MNISTDataset(Dataset):
    def __init__(self, train=True, download=True, transform=None):
        self.dataset = MNIST(
            root="./data/mnist",
            train=train,
            download=download,
            transform=transform or transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# -------------------------
# CIFAR-100
# -------------------------
@DatasetRegistry.register("CIFAR100")
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


# -------------------------
# Oxford-IIIT Pet
# -------------------------
@DatasetRegistry.register("OxfordPet")
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


# -------------------------
# Penn-Fudan Pedestrian
# -------------------------
@DatasetRegistry.register("PennFudanPed")
class PennFudanPedDataset(Dataset):
    def __init__(self, transforms=None, download=True):
        self.dataset = PennFudanPed(
            root="./data/pennfudan",
            download=download,
            transforms=transforms
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# -------------------------
# PatchCamelyon
# -------------------------
@DatasetRegistry.register("PatchCamelyon")
class PatchCamelyonDataset(Dataset):
    def __init__(self, train=True, transform=None):
        from torchvision.datasets import PatchCamelyon  # available in torchvision >=0.12
        self.dataset = PatchCamelyon(
            root="./data/patch_camelyon",
            split="train" if train else "test",
            download=True,
            transform=transform or transforms.ToTensor()
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


# -------------------------
# UCI Adult Income
# -------------------------
@DatasetRegistry.register("UCIAdult")
class UCIAdultDataset(Dataset):
    """
    UCI Adult Census Income Dataset
    https://archive.ics.uci.edu/ml/datasets/adult
    """
    def __init__(self, csv_path="./data/adult.csv", test_size=0.2, random_state=42):
        # Expect CSV with column 'income' as target
        self.df = pd.read_csv(csv_path)
        self.features = self.df.drop(columns=["income"])
        self.targets = self.df["income"].apply(lambda x: 1 if x.strip() == ">50K" else 0)

        X_train, X_test, y_train, y_test = train_test_split(
            self.features.values, self.targets.values, test_size=test_size, random_state=random_state
        )

        self.train_data = list(zip(torch.tensor(X_train, dtype=torch.float32),
                                   torch.tensor(y_train, dtype=torch.long)))
        self.test_data = list(zip(torch.tensor(X_test, dtype=torch.float32),
                                  torch.tensor(y_test, dtype=torch.long)))

    def get_split(self, train=True):
        return self.train_data if train else self.test_data

    def __len__(self):
        return len(self.train_data) + len(self.test_data)

    def __getitem__(self, idx):
        # By default, treat idx as index into train + test combined
        if idx < len(self.train_data):
            return self.train_data[idx]
        else:
            return self.test_data[idx - len(self.train_data)]
