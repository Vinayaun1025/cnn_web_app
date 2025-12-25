import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.cnn import CarBikeCNN
from src.data.dataset import get_dataloader
from src.data.transforms import train_transform, val_test_transform
from src.config.config import *
from src.utils.logger import setup_logger

logger = setup_logger()

def train():
    logger.info("Training started")

    train_loader = get_dataloader("data/processed/train", train_transform, BATCH_SIZE)
    val_loader = get_dataloader("data/processed/val", val_test_transform, BATCH_SIZE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = CarBikeCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] started")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} completed | Avg Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info(f"Model saved at {MODEL_PATH}")
    logger.info("Training finished successfully")

if __name__ == "__main__":
    train()
