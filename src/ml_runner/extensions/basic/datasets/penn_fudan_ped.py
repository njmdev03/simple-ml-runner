# from torchvision.datasets import PennFudanPed
from torch.utils.data import Dataset as TorchDataset

from ml_runner.core.registries import Dataset


# @Dataset("PennFudanPed")
# class PennFudanPedDataset(TorchDataset):
#     def __init__(self, transforms=None, download=True):
#         self.dataset = PennFudanPed(
#             root="./data/pennfudan",
#             download=download,
#             transforms=transforms
#         )

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         return self.dataset[idx]
