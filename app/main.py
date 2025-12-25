from fastapi import FastAPI, UploadFile, File, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from src.inference.predictor import predict_image
import shutil, os

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

os.makedirs("temp", exist_ok=True)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = predict_image(file_path)

    return {
        "prediction": prediction
    }
