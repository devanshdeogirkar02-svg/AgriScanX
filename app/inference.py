"""
Inference pipeline for AgriScanX.

Loads the trained model and predicts
a Near-Infrared (NIR) image from an RGB input.
"""

import torch
import numpy as np

from app.model import EfficientUNetGenerator
from app.preprocess import preprocess_image
from configs.config import DEVICE, MODEL_PATH


class InferenceEngine:
    """
    Handles model loading and inference.
    """

    def __init__(self):

        self.model = EfficientUNetGenerator().to(DEVICE)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=DEVICE,
        )

        self.model.load_state_dict(checkpoint)

        self.model.eval()

    @torch.no_grad()
    def predict(self, image_path: str):

        

        original_image, input_tensor = preprocess_image(image_path)

        original_image = np.array(original_image)

        input_tensor = input_tensor.to(DEVICE)

        prediction = self.model(input_tensor)

        prediction = (
            prediction
            .squeeze(0)
            .squeeze(0)
            .cpu()
        )

        prediction = prediction * 0.5 + 0.5

        return original_image, prediction.numpy()