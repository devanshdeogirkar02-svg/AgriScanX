from app.model import EfficientUNetGenerator, PatchGAN

G = EfficientUNetGenerator(base_channels=40).to(DEVICE)
D = PatchGAN().to(DEVICE)