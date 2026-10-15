
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


BAND_NAMES = ("G", "R", "RE", "NIR")


def normalize_channel(arr):
    arr = arr.astype(np.float32)
    mn = arr.min()
    mx = arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def load_band(path):
    # Read a spectral TIFF without converting it into RGB.
    # The first band is used if the TIFF has multiple samples.
    import rasterio

    with rasterio.open(path) as src:
        band = src.read(1)

    return normalize_channel(band)


def build_pairs(dataset_root):
    root = Path(dataset_root)
    rgb_dir = root / "RGB"
    ms_dir = root / "Multispectral"

    if not rgb_dir.is_dir() or not ms_dir.is_dir():
        raise FileNotFoundError(
            "Expected RGB/ and Multispectral/ directories."
        )

    pairs = []

    for rgb_path in sorted(rgb_dir.iterdir()):
        if rgb_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        band_paths = []
        for band in BAND_NAMES:
            matches = [
                p for p in ms_dir.glob(rgb_path.stem + "_" + band + ".*")
                if p.suffix.lower() in (".tif", ".tiff")
            ]

            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one {band} TIFF for {rgb_path.name}; "
                    f"found {len(matches)}."
                )

            band_paths.append(matches[0])

        pairs.append((rgb_path, band_paths))

    if not pairs:
        raise RuntimeError("No complete RGB-MS image pairs found.")

    return pairs


class WeedyRiceCorrectedDataset(Dataset):
    def __init__(self, dataset_root, image_size=256):
        self.pairs = build_pairs(dataset_root)
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        rgb_path, band_paths = self.pairs[index]

        rgb = np.asarray(
            Image.open(rgb_path).convert("RGB"),
            dtype=np.float32
        )

        rgb = np.stack(
            [normalize_channel(rgb[:, :, c]) for c in range(3)],
            axis=0
        )

        ms = np.stack(
            [load_band(path) for path in band_paths],
            axis=0
        )

        rgb_t = torch.from_numpy(rgb.copy()).float()
        ms_t = torch.from_numpy(ms.copy()).float()

        size = (self.image_size, self.image_size)

        rgb_t = F.interpolate(
            rgb_t.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        ms_t = F.interpolate(
            ms_t.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        return rgb_t, ms_t
