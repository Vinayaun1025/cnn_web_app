from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.inference.predictor import predict_image, load_model

import shutil
import os

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# Templates & static files
# -------------------------------------------------
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# -------------------------------------------------
# Temp directory (Azure writable)
# -------------------------------------------------
TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

# -------------------------------------------------
# Global model (lazy loaded once)
# -------------------------------------------------
model = None


# -------------------------------------------------
# Startup event (runs once per container)
# -------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model
    model = load_model()
    print("CNN model loaded successfully")


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = os.path.join(TEMP_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run prediction using preloaded model
    prediction = predict_image(file_path, model)

    return {
        "prediction": prediction
    }
