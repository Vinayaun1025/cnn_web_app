from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from src.inference.predictor import (
    load_model,
    predict_image_from_bytes
)

app = FastAPI()

# -------------------------------------------------
# Templates & static files
# -------------------------------------------------
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# -------------------------------------------------
# Global model (initialized at startup)
# -------------------------------------------------
model = None

# -------------------------------------------------
# Startup event (Azure-safe)
# -------------------------------------------------
@app.on_event("startup")
def startup_event():
    global model
    try:
        model = load_model()
        print("Model loaded successfully")
    except Exception as e:
        print("Model load failed:", e)
        model = None

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
    if model is None:
        return {"error": "Model not loaded on server"}

    image_bytes = await file.read()

    if not image_bytes:
        return {"error": "Empty file uploaded"}

    prediction = predict_image_from_bytes(image_bytes, model)
    return {"prediction": prediction}
