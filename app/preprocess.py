"""
Image preprocessing utilities for AgriScanX.

This module prepares RGB images before they are
passed to the trained neural network.
"""

from PIL import Image
import torchvision.transforms as transforms

from configs.config import IMAGE_SIZE


transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
        ),
    ]
)


def preprocess_image(image_path: str):
    """
    Loads an RGB image and converts it into
    a normalized tensor.
    """

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        tensor = transform(img).unsqueeze(0)

    return img, tensor