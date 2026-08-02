"""
Evaluation script for AgriScanX.

Computes image quality metrics between
predicted NIR images and ground truth NIR images.

Metrics:
- PSNR
- SSIM
- MSE



import numpy as np

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity,
    mean_squared_error,
)


def calculate_psnr(
    ground_truth,
    prediction,
):
    
    Calculate Peak Signal-to-Noise Ratio.
    

    return peak_signal_noise_ratio(
        ground_truth,
        prediction,
        data_range=1.0,
    )


def calculate_ssim(
    ground_truth,
    prediction,
):
    
    Calculate Structural Similarity Index.
    

    return structural_similarity(
        ground_truth,
        prediction,
        data_range=1.0,
    )



def calculate_mse(
    ground_truth,
    prediction,
):
    
    Calculate Mean Squared Error.
    

    return mean_squared_error(
        ground_truth,
        prediction,
    )




Dataset was used during research but is not included in this repository.
"""


# TODO:
# 1. Load trained model
# 2. Load evaluation dataset
# 3. Run inference
# 4. Compute PSNR
# 5. Compute SSIM
# 6. Compute MSE
# 7. Report average metrics