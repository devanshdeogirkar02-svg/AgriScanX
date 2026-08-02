"""
Configuration settings for AgriScanX.

This file stores project-wide constants used for
training, inference, and data processing.
"""

from pathlib import Path
import torch

# ==========================================================
# Project Directories
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLE_DIR = PROJECT_ROOT / "sample_images"

# Create directories automatically
MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ==========================================================
# Dataset
# ==========================================================

IMAGE_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 2

PAIRED_DIR = DATA_DIR / "rgb2nir" / "train"

# ==========================================================
# Model
# ==========================================================

BASE_CHANNELS = 40

MODEL_PATH = MODEL_DIR / "finetuned_epoch40 (1).pth"

# ==========================================================
# Training
# ==========================================================

LEARNING_RATE = 2e-4

EPOCHS = 100

LAMBDA_L1 = 100

LAMBDA_NDVI = 10

# ==========================================================
# Device
# ==========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)