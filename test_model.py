
import argparse
from pathlib import Path

import torch

from rgb2ms_cbam_model import RGB2MS_UNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="models/best_model.pth"
    )
    args = parser.parse_args()

    path = Path(args.checkpoint)

    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "The large CBAM checkpoint is distributed separately."
        )

    model = RGB2MS_UNet(in_ch=3, out_ch=4, base=64)

    state = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state, strict=True)
    model.eval()

    print("Checkpoint loaded successfully!")

    with torch.inference_mode():
        x = torch.randn(1, 3, 64, 64)
        output = model(x)

    assert output.shape == (1, 4, 64, 64)

    print("Input shape:", tuple(x.shape))
    print("Output shape:", tuple(output.shape))
    print("Model test PASSED!")


if __name__ == "__main__":
    main()
