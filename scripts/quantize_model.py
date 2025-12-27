import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.abspath("."))

from src.models.cnn import CarBikeCNN

DEVICE = torch.device("cpu")

ORIGINAL_MODEL = "models/best_model.pth"
QUANTIZED_MODEL = "models/best_model_quantized.pth"

model = CarBikeCNN()
model.load_state_dict(torch.load(ORIGINAL_MODEL, map_location=DEVICE))
model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

#  SAVE FULL MODEL (important)
torch.save(quantized_model, QUANTIZED_MODEL)

print("Quantized model saved correctly")
