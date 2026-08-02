"""
Main entry point for AgriScanX.

Example:
    python app/main.py sample_images/input/leaf.jpg
"""

from pathlib import Path
import sys

from app.inference import InferenceEngine
from app.ndvi import (
    calculate_ndvi,
    colorize_ndvi,
    vegetation_mask,
)
from app.utils import (
    save_image,
    save_numpy,
    normalize_to_uint8,
)
from configs.config import OUTPUT_DIR


def main():

    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python app/main.py <image_path>"
        )
        return

    image_path = Path(sys.argv[1])

    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return

    print("=" * 50)
    print("AgriScanX")
    print("=" * 50)

    print("Loading model...")

    engine = InferenceEngine()

    print("Running inference...")

    rgb_image, predicted_nir = engine.predict(str(image_path))

    ndvi = calculate_ndvi(
        rgb_image,
        predicted_nir,
    )

    ndvi_color = colorize_ndvi(ndvi)

    mask = vegetation_mask(ndvi)

    save_image(
        normalize_to_uint8(predicted_nir),
        OUTPUT_DIR / "predicted_nir.png",
    )

    save_image(
        ndvi_color,
        OUTPUT_DIR / "ndvi_map.png",
    )

    save_image(
        mask,
        OUTPUT_DIR / "vegetation_mask.png",
    )

    save_numpy(
        ndvi,
        OUTPUT_DIR / "ndvi.npy",
    )

    print()

    print("Prediction completed successfully.")

    print(f"NIR Image      : {OUTPUT_DIR/'predicted_nir.png'}")
    print(f"NDVI Map       : {OUTPUT_DIR/'ndvi_map.png'}")
    print(f"Vegetation Mask: {OUTPUT_DIR/'vegetation_mask.png'}")

    print("=" * 50)


if __name__ == "__main__":
    main()