"""
Dataset utilities for AgriScanX.

Provides:
- PairedDataset
- create_dataloader()
"""

import glob
import random

from PIL import Image

import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torch.utils.data import Dataset, DataLoader

from configs.config import (
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    PAIRED_DIR,
)



class PairedDataset(Dataset):
    """
    Dataset for paired RGB-NIR training images.

    Each image contains:
    - Left half : RGB image
    - Right half: NIR image
    """

      
    def __init__(self, image_paths: list[str]):
        self.image_paths = image_paths
        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        w = img.width // 2
        rgb = img.crop((0, 0, w, img.height))
        nir = img.crop((w, 0, img.width, img.height))

        rgb = self.transform(rgb)
        nir = transforms.Grayscale()(self.transform(nir))

        # Augmentations
        if random.random() > 0.5:
            rgb = TF.hflip(rgb)
            nir = TF.hflip(nir)
        if random.random() > 0.5:
            rgb = TF.vflip(rgb)
            nir = TF.vflip(nir)
        if random.random() > 0.6:
            angle = random.choice([90, 180, 270])
            rgb = TF.rotate(rgb, angle)
            nir = TF.rotate(nir, angle)

        return rgb, nir




def create_dataloader() -> DataLoader:
    """
    Creates a PyTorch DataLoader for paired RGB-NIR images.
    """

    image_paths = sorted(glob.glob(str(PAIRED_DIR / "*.png")))
    image_paths += sorted(glob.glob(str(PAIRED_DIR / "*.jpg")))

    if not image_paths:
        raise FileNotFoundError(
            f"No training images found in {PAIRED_DIR}"
        )

    dataset = PairedDataset(image_paths)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
