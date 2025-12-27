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
# Load model ONCE (global)
# -------------------------------------------------
model = load_model()

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
    image_bytes = await file.read()

    if not image_bytes:
        return {"error": "Empty file uploaded"}

    prediction = predict_image_from_bytes(image_bytes, model)

    return {"prediction": prediction}
