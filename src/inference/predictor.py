import os
import io
import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn import CarBikeCNN

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_PATH = os.path.join("models", "best_model.pth")
DEVICE = torch.device("cpu")

# -------------------------------------------------
# Image preprocessing
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
# Load model ONCE (safe for Flask/Azure)
# -------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = CarBikeCNN()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully")
    return model


# -------------------------------------------------
# Predict from image bytes (Flask upload)
# -------------------------------------------------
def predict_image_from_bytes(image_bytes: bytes, model) -> str:
    if model is None:
        return "Model not loaded"

    try:
        # Validate input bytes
        if image_bytes is None or len(image_bytes) == 0:
            return "Empty image uploaded"

        # Load image safely
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image = transform(image).unsqueeze(0).to(DEVICE)

        # Inference
        with torch.no_grad():
            logits = model(image).view(-1)
            prob = torch.sigmoid(logits).item()

        # Final decision
        if prob >= 0.5:
            return "Car 🚗"
        else:
            return "Bike 🏍️"

    except Exception as e:
        print("Prediction error:", str(e))
        return "Prediction failed"
