
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from rgb2ms_cbam_model import RGB2MS_UNet


def preprocess(image_path, size=256):
    image = np.asarray(
        Image.open(image_path).convert("RGB"),
        dtype=np.float32
    )

    # Match the original training notebook:
    # independent min-max normalization per RGB channel.
    image = image.transpose(2, 0, 1)

    for c in range(3):
        mn = image[c].min()
        mx = image[c].max()
        image[c] = (image[c] - mn) / (mx - mn + 1e-8)

    tensor = torch.from_numpy(image.copy())

    tensor = F.interpolate(
        tensor.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )

    return tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pth"
    )
    parser.add_argument("--out", default="outputs/prediction")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    if args.size <= 0 or args.size % 16 != 0:
        parser.error("--size must be a positive multiple of 16")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}"
        )

    model = RGB2MS_UNet(in_ch=3, out_ch=4, base=64)

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    x = preprocess(args.image, args.size)

    with torch.inference_mode():
        prediction = model(x)[0].cpu().numpy()

    print("Input shape:", tuple(x.shape))
    print("Prediction shape:", prediction.shape)

    np.save(output_dir / "prediction.npy", prediction)

    for i in range(prediction.shape[0]):
        channel = np.clip(prediction[i], 0, 1)
        image = Image.fromarray(
            np.round(channel * 255).astype(np.uint8)
        )
        image.save(output_dir / f"channel_{i}.png")

    print("Saved predictions to:", output_dir)
    print(
        "WARNING: Experimental checkpoint. "
        "Channels are not validated spectral measurements."
    )


if __name__ == "__main__":
    main()
