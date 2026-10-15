# AgriScanX

**AI-Powered RGB-to-NIR Reconstruction and Experimental Multispectral Imaging for Precision Agriculture**

AgriScanX is an experimental deep-learning project that explores predicting near-infrared (NIR) and multispectral-like image channels from standard RGB photographs.

The project includes a PyTorch-based inference pipeline, computer-vision preprocessing, experimental vegetation analysis components, and architectures intended for future edge deployment.

## Project Overview

Multispectral cameras can capture information beyond ordinary RGB images, including NIR and Red Edge bands. AgriScanX investigates whether machine-learning models can estimate useful image representations from RGB inputs.

The project currently provides two separate experimental model pipelines.

## Model Architectures

| Model | Architecture | Output | Status |
|---|---|---|---|
| EfficientUNetGenerator | Compact U-Net-based generator | One estimated NIR channel | Checkpoint loading and RGB inference tested |
| RGB2MS_UNet | U-Net with CBAM attention | Four experimental output channels | Checkpoint loading and inference tested separately |

These models use different checkpoints and must not be interchanged.

## System Architecture

**Existing RGB-to-NIR application**

```text
RGB Image
    |
    v
Image Preprocessing
    |
    v
EfficientUNetGenerator (PyTorch)
    |
    v
Estimated NIR Image
    |
    v
Experimental Visualization / Vegetation Analysis
```

**Experimental CBAM pipeline**

```text
RGB Image
    |
    v
CBAM Model Preprocessing
    |
    v
RGB2MS_UNet
    |
    v
Four Experimental Output Channels
    |
    v
Saved Images and NumPy Prediction
```

## Project Structure

```text
AgriScanX/
|-- app/                     # Existing application
|-- configs/                 # Configuration
|-- models/                  # Compact trained checkpoint
|-- dataset_corrected.py     # Corrected four-band dataset pairing
|-- rgb2ms_cbam_model.py     # Experimental CBAM model
|-- predict.py               # CBAM inference script
|-- test_model.py            # CBAM checkpoint test
|-- requirements.txt
|-- README.md
```

Additional folders and research files may be present in the repository.

## Technology Stack

- Python
- PyTorch
- NumPy
- Pillow and OpenCV
- FastAPI application components
- Git and GitHub

Raspberry Pi deployment is a future development goal rather than a verified deployment.

## Installation

Clone the repository:

```bash
git clone https://github.com/devanshdeogirkar02-svg/AgriScanX.git
cd AgriScanX
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install project dependencies:

```bash
pip install -r requirements.txt
```

If required by your environment, install the appropriate PyTorch build using the official PyTorch installation instructions.

## RGB-to-NIR Inference

The compact checkpoint is included in:

```text
models/finetuned_epoch40 (1).pth
```

The application's inference engine reads its checkpoint path from `configs/config.py`.

Example Python usage:

```python
from app.inference import InferenceEngine

engine = InferenceEngine()

rgb, nir = engine.predict("path/to/rgb_image.jpg")

print("RGB:", rgb.shape)
print("Estimated NIR:", nir.shape)
```

The existing inference pipeline has been tested with a 960 × 1280 RGB image and generated a 256 × 256 estimated NIR output.

## Experimental CBAM Model

The separate CBAM checkpoint, `best_model.pth`, is too large to store in ordinary GitHub Git history.

It is available as a release asset in the companion repository:

[Download experimental CBAM checkpoint](https://github.com/shreya-1310/Agriscanx/releases/tag/v0.1.0)

Place the downloaded checkpoint at:

```text
models/best_model.pth
```

Test the checkpoint using:

```bash
python test_model.py
```

Run inference:

```bash
python predict.py --image path/to/rgb_image.jpg --checkpoint models/best_model.pth --out outputs/test_prediction
```

The script generates four experimental channel images and a NumPy prediction array.

## Dataset

Research experiments use the WeedyRice-RGBMS-DB dataset.

The `dataset_corrected.py` loader explicitly pairs RGB images with separate Green, Red, Red Edge and NIR TIFF files.

The original CBAM training notebook used a problematic band-pairing method. Therefore, the legacy CBAM model's four outputs cannot be assumed to represent properly learned, validated spectral bands.

## Validation and Limitations

- Compact model checkpoint loading: **Tested**
- Compact RGB-to-NIR inference: **Tested**
- CBAM checkpoint loading and image inference: **Tested separately**
- Corrected dataset loader: **Sample tested**
- Scientifically validated spectral reconstruction: **Not established**
- Reliable NDVI-based crop-health classification: **Not established**
- Raspberry Pi deployment: **Not verified**

Estimated image values should not be treated as calibrated spectral reflectance or used to make agricultural decisions without further scientific validation.

## Future Improvements

- Retrain using correctly paired multispectral bands
- Evaluate with held-out ground-truth data
- Add quantitative reconstruction metrics
- Validate vegetation index calculations
- Expand API integration and testing
- Optimize models for Raspberry Pi deployment
- Develop an interactive dashboard
- Add batch inference and automated tests

## Author

**Devansh Deogirkar**

Electronics and Communication Engineering Student

Interests: Artificial Intelligence, Computer Vision, Edge AI, Backend Development and System Design.