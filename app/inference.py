from app.model import EfficientUNetGenerator
from app.preprocess import preprocess_image
from configs.config import DEVICE, MODEL_PATH, BASE_CHANNELS

import torch


def load_model():

    model = EfficientUNetGenerator(
        base_channels=BASE_CHANNELS
    ).to(DEVICE)

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(state_dict)

    model.eval()

    return model