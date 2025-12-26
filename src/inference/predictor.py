import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn import CarBikeCNN

# -------------------------------------------------
# Constants
# -------------------------------------------------
MODEL_PATH = "models/best_model.pth"

# Azure App Service = CPU only
DEVICE = torch.device("cpu")

# -------------------------------------------------
# Image transforms (safe at import)
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# -------------------------------------------------
# Load model ONCE at startup
# -------------------------------------------------
def load_model():
    model = CarBikeCNN()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model


# -------------------------------------------------
# Predict using preloaded model
# -------------------------------------------------
def predict_image(image_path: str, model) -> str:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        prob = torch.sigmoid(logits).item()

    return "Car 🚗" if prob >= 0.5 else "Bike 🏍️"
