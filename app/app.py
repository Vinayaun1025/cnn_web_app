from flask import Flask, render_template, request, jsonify
from src.inference.predictor import load_model, predict_image_from_bytes
import traceback

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# -------------------------------------------------
# Load model ONCE (global, Azure-safe)
# -------------------------------------------------
model = None

def init_model():
    global model
    try:
        model = load_model()
        print("Model loaded successfully")
    except Exception as e:
        print("Model loading failed")
        traceback.print_exc()
        model = None

init_model()

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"prediction": "Model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"prediction": "No file uploaded"}), 400

    file = request.files["file"]
    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"prediction": "Empty image"}), 400

    try:
        prediction = predict_image_from_bytes(image_bytes, model)
        return jsonify({"prediction": prediction})

    except Exception as e:
        print("Prediction error")
        traceback.print_exc()
        return jsonify({"prediction": "Prediction failed"}), 500


# -------------------------------------------------
# Local run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
