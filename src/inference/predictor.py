import os
import io
import torch
from PIL import Image
from torchvision import transforms

DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model_quantized.pth")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

def load_model():
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    model.eval()
    return model

def predict_image_from_bytes(image_bytes: bytes, model):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image)
        prob = torch.sigmoid(logits).item()

    return "Car 🚗" if prob >= 0.5 else "Bike 🏍️"
