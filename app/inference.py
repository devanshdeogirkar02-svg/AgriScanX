from app.model import EfficientUNetGenerator

model = EfficientUNetGenerator(base_channels=40)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()