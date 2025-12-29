import os
import traceback
from flask import Flask, render_template, request, jsonify

from src.inference.predictor import load_model, predict_image_from_bytes

# -------------------------------------------------
# Flask App
# -------------------------------------------------
app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Limit upload size to 5MB (recommended)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# -------------------------------------------------
def load_model():
    print("Looking for model at:",MODEL_PATH)
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = torch.load(MODEL_PATH, map_location='cpu')
    model.eval()
    return model


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"prediction": "Model not available"}), 500

    if "file" not in request.files:
        return jsonify({"prediction": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"prediction": "Empty filename"}), 400

    # Basic image validation
    if not file.mimetype.startswith("image/"):
        return jsonify({"prediction": "Invalid file type"}), 400

    image_bytes = file.read()

    if not image_bytes:
        return jsonify({"prediction": "Empty image"}), 400

    try:
        prediction = predict_image_from_bytes(image_bytes, model)
        return jsonify({"prediction": prediction})

    except Exception:
        print("Prediction error")
        traceback.print_exc()
        return jsonify({"prediction": "Prediction failed"}), 500


# -------------------------------------------------
# Local run (NOT used on Azure)
# -------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Flask app on port {port}")
    app.run(host="0.0.0.0", port=port, debug=True)
