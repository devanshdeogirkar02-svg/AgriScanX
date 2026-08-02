"""
Utility functions for AgriScanX.
"""

from pathlib import Path
import cv2
import numpy as np


def ensure_dir(path: Path):
    """
    Create directory if it does not exist.
    """
    path.mkdir(parents=True, exist_ok=True)


def save_image(
    image: np.ndarray,
    output_path: Path,
):
    """
    Save image to disk.
    """
    ensure_dir(output_path.parent)

    cv2.imwrite(
        str(output_path),
        image,
    )


def save_numpy(
    array: np.ndarray,
    output_path: Path,
):
    """
    Save numpy array to disk.
    """
    ensure_dir(output_path.parent)

    np.save(
        output_path,
        array,
    )


def normalize_to_uint8(
    image: np.ndarray,
):
    """
    Convert float image to uint8.
    """

    image = image - image.min()

    if image.max() > 0:
        image = image / image.max()

    image = image * 255

    return image.astype(np.uint8)