import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.models.cnn import CarBikeCNN
from src.data.dataset import get_dataloader
from src.data.transforms import val_test_transform
from src.config.config import *

def evaluate():
    print("Starting evaluation...")

    loader = get_dataloader(
        "data/processed/test",
        val_test_transform,
        BATCH_SIZE,
        shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model safely
    model = CarBikeCNN().to(device)
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.view(-1).cpu().numpy()  # ensure shape + numpy

            logits = model(images).view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n Accuracy: {acc:.4f}")

    print("\n Classification Report:")
    print(classification_report(
        all_labels,
        all_preds,
        target_names=["Bike 🏍️", "Car 🚗"]
    ))

    print("\n Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
