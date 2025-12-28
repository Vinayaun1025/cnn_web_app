import io
import torch
from PIL import Image
from torchvision import transforms
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_quantized.pth")

DEVICE = torch.device("cpu")

# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# -------------------------------------------------
# Load model ONCE
def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

# -------------------------------------------------
# Predict from BYTES (not path)
def predict_image_from_bytes(image_bytes: bytes, model) -> str:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

        # Ensure tensor
        if isinstance(output, (list, tuple)):
            output = output[0]

        prob = torch.sigmoid(output).item()

    return "Car 🚗" if prob >= 0.5 else "Bike 🏍️"
