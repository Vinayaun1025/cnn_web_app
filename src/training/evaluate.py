import torch
from sklearn.metrics import accuracy_score
from src.models.cnn import CarBikeCNN
from src.data.dataset import get_dataloader
from src.data.transforms import val_test_transform
from src.config.config import *

def evaluate():
    loader = get_dataloader("data/processed/test", val_test_transform, BATCH_SIZE, False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CarBikeCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    preds, labels_all = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images)).cpu()
            preds.extend((outputs > 0.5).int().numpy())
            labels_all.extend(labels.numpy())

    print("Accuracy:", accuracy_score(labels_all, preds))

if __name__ == "__main__":
    evaluate()
