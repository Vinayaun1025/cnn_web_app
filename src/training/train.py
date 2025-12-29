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
    logger.info("🚀 Training started")

    # -------------------------------
    # Data loaders
    # -------------------------------
    train_loader = get_dataloader(
        "data/processed/train",
        train_transform,
        BATCH_SIZE,
        shuffle=True
    )

    val_loader = get_dataloader(
        "data/processed/val",
        val_test_transform,
        BATCH_SIZE,
        shuffle=False
    )

    # -------------------------------
    # Device
    # -------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # -------------------------------
    # Model
    # -------------------------------
    model = CarBikeCNN().to(device)

    # IMPORTANT: Use BCEWithLogitsLoss (NO sigmoid here)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4   # helps generalization
    )

    best_val_loss = float("inf")

    # -------------------------------
    # Training loop
    # -------------------------------
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        logger.info(f"Epoch [{epoch+1}/{EPOCHS}] started")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.float().to(device).view(-1)  # FIX SHAPE

            optimizer.zero_grad()

            logits = model(images).view(-1)  # FIX SHAPE
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        avg_train_loss = train_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")

        # -------------------------------
        # Validation
        # -------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.float().to(device).view(-1)

                logits = model(images).view(-1)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        # -------------------------------
        # Save BEST model only
        # -------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(
                model.state_dict(),
                MODEL_PATH
            )
            logger.info(f"💾 Best model saved at {MODEL_PATH}")

    logger.info("Training finished successfully")


if __name__ == "__main__":
    train()
