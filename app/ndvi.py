"""
NDVI computation utilities for AgriScanX.

This module provides functions to:
- Calculate NDVI from RGB and predicted NIR images.
- Generate a colored NDVI visualization.
- Create a binary vegetation mask.
"""

import cv2
import numpy as np


def calculate_ndvi(
    rgb_image: np.ndarray,
    nir_image: np.ndarray,
) -> np.ndarray:
    """
    Calculate the Normalized Difference Vegetation Index (NDVI).

    Parameters
    ----------
    rgb_image : np.ndarray
        Original RGB image.

    nir_image : np.ndarray
        Predicted NIR image.

    Returns
    -------
    np.ndarray
        NDVI values in the range [-1, 1].
    """

    # Resize RGB image to match predicted NIR dimensions
    height, width = nir_image.shape
    rgb = cv2.resize(rgb_image, (width, height))

    # Extract Red channel (PIL loads images in RGB format)
    red = rgb[:, :, 0].astype(np.float32) / 255.0
    nir = nir_image.astype(np.float32)

    # Avoid division by zero
    denominator = nir + red
    denominator[denominator == 0] = 1e-6

    # NDVI calculation
    ndvi = (nir - red) / denominator

    return np.clip(ndvi, -1.0, 1.0)


def colorize_ndvi(
    ndvi: np.ndarray,
) -> np.ndarray:
    """
    Create a colored NDVI visualization.

    Parameters
    ----------
    ndvi : np.ndarray

    Returns
    -------
    np.ndarray
        Colorized NDVI image.
    """

    normalized = ((ndvi + 1.0) * 127.5).astype(np.uint8)

    colored = cv2.applyColorMap(
        normalized,
        cv2.COLORMAP_SUMMER,
    )

    return colored


def vegetation_mask(
    ndvi: np.ndarray,
    threshold: float = 0.30,
) -> np.ndarray:
    """
    Generate a binary vegetation mask from NDVI.

    Parameters
    ----------
    ndvi : np.ndarray

    threshold : float
        NDVI threshold above which vegetation is detected.

    Returns
    -------
    np.ndarray
        Binary vegetation mask.
    """

    mask = (ndvi > threshold).astype(np.uint8)

    return mask * 255