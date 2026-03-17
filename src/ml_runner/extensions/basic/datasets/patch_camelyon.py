# from torchvision.datasets import PatchCamelyon
from torchvision import transforms
from torch.utils.data import Dataset as TorchDataset

from ml_runner.core.registries import Dataset


# @Dataset("PatchCamelyon")
# class PatchCamelyonDataset(TorchDataset):
#     def __init__(self, train=True, transform=None):
#         self.dataset = PatchCamelyon(
#             root="./data/patch_camelyon",
#             split="train" if train else "test",
#             download=True,
#             transform=transform or transforms.ToTensor()
#         )

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]