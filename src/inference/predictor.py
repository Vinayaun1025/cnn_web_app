import torch
from PIL import Image
from torchvision import transforms
from src.models.cnn import CarBikeCNN

MODEL_PATH = "models/best_model.pth"

# Load model ONCE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CarBikeCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def predict_image(image_path: str) -> str:
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image)
        prob = torch.sigmoid(logits).item()

    # IMPORTANT: threshold logic
    if prob >= 0.5:
        return "Car 🚗"
    else:
        return "Bike 🏍️"
