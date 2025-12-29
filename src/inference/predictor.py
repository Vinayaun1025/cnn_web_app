import os
import io
import torch
from PIL import Image
from torchvision import transforms

from src.models.cnn import CarBikeCNN

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_PATH = os.path.join("models", "best_model_quantized.pth")
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
# Load model safely (called once)
# -------------------------------------------------
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    model = CarBikeCNN()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)

    # Handle quantized vs normal model
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully")
    return model

# -------------------------------------------------
# Predict from image bytes (Flask upload)
# -------------------------------------------------
def predict_image_from_bytes(image_bytes:bytes, model) -> str:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = model(image)
            print("Debug logits:", logits)
        
        
        prob = torch.sigmoid(logits).item() 
        print("Debug prob:", prob)
        return "car" if prob >= 0.5 else "bike"

    except Exception as e:
        print("Error during prediction",str(e))
        return f"Prediction failed: {str(e)}"