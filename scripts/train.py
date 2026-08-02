"""
Training script for AgriScanX.

This script trains the EfficientUNetGenerator model
using paired RGB-NIR images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from configs.config import (
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    LAMBDA_L1,
    LAMBDA_NDVI,
)

from app.model import (
    EfficientUNetGenerator,
    PatchGAN,
)

from scripts.create_dataset import create_dataloader



def ndvi_loss(fake_nir, rgb):

    red = rgb[:, 0:1]

    ndvi = (fake_nir - red) / (fake_nir + red + 1e-6)

    target = torch.clamp(ndvi, -0.2, 0.9)

    return nn.L1Loss()(ndvi, target)



generator = EfficientUNetGenerator().to(DEVICE)
discriminator = PatchGAN().to(DEVICE)



gan_loss = nn.MSELoss()
l1_loss = nn.L1Loss()



optimizer_g = optim.Adam(
    generator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)
optimizer_d = optim.Adam(
    discriminator.parameters(),
    lr=LEARNING_RATE,
    betas=(0.5, 0.999),
)



train_loader = create_dataloader()