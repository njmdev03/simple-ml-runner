"""
Penn-Fudan Pedestrian Dataset utilities for Object Detection.
Downloads the dataset automatically and wraps it in a PyTorch Dataset.

Dataset: https://www.cis.upenn.edu/~jshi/ped_html/
- 170 images of pedestrians
- Pixel-level instance segmentation masks
- Also suitable as a bounding-box detection benchmark
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
import torchvision.transforms.v2 as T


def default_transforms():
    """Standard Faster R-CNN image preprocessing: resize longest side to 800px."""
    return T.Compose([
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
    ])


class PennFudanDataset(Dataset):
    """Penn-Fudan Pedestrian Detection Dataset.

    Each item returns:
        image: Tensor [C, H, W] float32 in [0, 1]
        target: dict with 'boxes' [N, 4], 'labels' [N], 'image_id', 'area', 'iscrowd'
    """

    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = Path(root).resolve()  # absolute so DataLoader workers don't lose the path
        self.transform = transform if transform is not None else default_transforms()

        # Auto-download if the dataset isn't present yet
        if not (self.root / "PNGImages").exists():
            download_penn_fudan(dest=str(self.root.parent))

        all_imgs = sorted((self.root / "PNGImages").glob("*.png"))
        all_masks = sorted((self.root / "PedMasks").glob("*.png"))

        # 80/20 train/val split
        split_idx = int(len(all_imgs) * 0.8)
        if train:
            self.imgs = all_imgs[:split_idx]
            self.masks = all_masks[:split_idx]
        else:
            self.imgs = all_imgs[split_idx:]
            self.masks = all_masks[split_idx:]

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        mask = np.array(mask)
        obj_ids = np.unique(mask)[1:]  # skip background (0)

        masks_binary = mask == obj_ids[:, None, None]

        boxes = []
        for m in masks_binary:
            pos = np.where(m)
            xmin, xmax = pos[1].min(), pos[1].max()
            ymin, ymax = pos[0].min(), pos[0].max()
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)  # class 1 = pedestrian

        from torchvision import tv_tensors
        boxes = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=img.size[::-1])

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 else torch.zeros(0)
        iscrowd = torch.zeros(len(labels), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transform is not None:
             img, target = self.transform(img, target)
        else:
             img = F.to_tensor(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


def download_penn_fudan(dest: str = "./data"):
    """Download and extract the Penn-Fudan dataset."""
    import urllib.request
    import zipfile

    dest = Path(dest)
    (dest / "PennFudanPed").mkdir(parents=True, exist_ok=True)

    url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
    zip_path = dest / "PennFudanPed.zip"
    extracted = dest / "PennFudanPed" / "PNGImages"

    if extracted.exists():
        print("Penn-Fudan already downloaded.")
        return dest / "PennFudanPed"

    print(f"Downloading Penn-Fudan dataset to {zip_path}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(dest)

    zip_path.unlink()
    print("Done.")
    return dest / "PennFudanPed"
