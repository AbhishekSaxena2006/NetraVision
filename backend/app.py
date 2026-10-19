import os
import io
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Optional imports for real model handling
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TF_AVAILABLE = True
except Exception as e:
    logging.warning(f"TensorFlow not available or failed to import: {e}")
    TF_AVAILABLE = False

try:
    from PIL import Image
except Exception as e:
    logging.error(f"Pillow is required for image handling: {e}")
    raise

app = Flask(__name__)
CORS(app)

# -------------------- Configuration --------------------
# These can be overridden via environment variables as needed.
MODEL_PATH = os.getenv("MODEL_PATH", "dr_model.keras")  # relative to project root
INPUT_HEIGHT = int(os.getenv("INPUT_HEIGHT", "224"))
INPUT_WIDTH = int(os.getenv("INPUT_WIDTH", "224"))
INPUT_SIZE: Tuple[int, int] = (INPUT_HEIGHT, INPUT_WIDTH)
# preprocessing mode: "normalize" -> divide by 255, "none" -> raw pixels
PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "normalize")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# -------------------- Model Loading --------------------
model = None
MOCK_MODE = False

def load_cnn_model() -> None:
    global model, MOCK_MODE
    if not TF_AVAILABLE:
        MOCK_MODE = True
        logging.info("TensorFlow unavailable – entering mock mode.")
        return
    model_path = Path(MODEL_PATH)
    if not model_path.is_file():
        MOCK_MODE = True
        logging.info(f"Model file not found at {model_path.resolve()}. Using mock mode.")
        return
    try:
        model = load_model(str(model_path))
        MOCK_MODE = False
        logging.info(f"Model loaded successfully from {model_path.resolve()}")
    except Exception as exc:
        MOCK_MODE = True
        logging.error(f"Failed to load model: {exc}. Switching to mock mode.")

load_cnn_model()

# -------------------- Helpers --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Resize, convert to RGB and apply preprocessing.
    Returns a NumPy array ready for model prediction.
    """
    with Image.open(io.BytesIO(image_bytes)) as img:
        img = img.convert("RGB")
        img = img.resize(INPUT_SIZE)
        arr = np.asarray(img, dtype=np.float32)
        if PREPROCESS_MODE == "normalize":
            arr = arr / 255.0
        # Additional preprocessing steps can be added here.
    return arr

def mock_prediction() -> Tuple[int, float, Dict[str, float]]:
    """Generate a deterministic mock prediction using Dirichlet distribution.
    Returns (predicted_class, confidence, scores_dict).
    """
    probs = np.random.dirichlet(np.ones(5), size=1)[0]
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])
    scores = {str(i): float(p) for i, p in enumerate(probs)}
    return pred_class, confidence, scores

CLASS_NAMES = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR",
}

# -------------------- Routes --------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "model_path": str(Path(MODEL_PATH).resolve()),
        "input_size": INPUT_SIZE,
        "preprocess": PREPROCESS_MODE,
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    # Validate request contains file
    if "image" not in request.files:
        return jsonify({"error": "No image file part in the request."}), 400
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file."}), 400
    if not allowed_file(file.filename):
        return (
            jsonify({"error": f"File type not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}),
            400,
        )
    # Flask already limits request size via MAX_CONTENT_LENGTH; additional check for safety
    file.seek(0, os.SEEK_END)
    file_length = file.tell()
    file.seek(0)
    if file_length > MAX_CONTENT_LENGTH:
        return jsonify({"error": "File exceeds maximum size of 10 MB."}), 400

    # Read file bytes
    try:
        image_bytes = file.read()
    except Exception as exc:
        return jsonify({"error": f"Failed to read image: {exc}"}), 500

    if MOCK_MODE:
        pred_class, confidence, scores = mock_prediction()
    else:
        try:
            img_array = preprocess_image(image_bytes)
            # Model expects batch dimension
            batch = np.expand_dims(img_array, axis=0)
            preds = model.predict(batch)
            probs = preds[0]
            # Ensure we have exactly 5 classes
            if probs.shape[0] != 5:
                raise ValueError(f"Model output dimension {probs.shape[0]} != 5")
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            scores = {str(i): float(p) for i, p in enumerate(probs)}
        except Exception as exc:
            logging.exception("Prediction failed")
            return jsonify({"error": f"Prediction error: {exc}"}), 500

    response = {
        "predicted_class": pred_class,
        "class_name": CLASS_NAMES.get(pred_class, "Unknown"),
        "confidence": confidence,
        "scores": scores,
        "mock_mode": MOCK_MODE,
    }
    return jsonify(response)

# -------------------- Main --------------------
if __name__ == "__main__":
    # Use port from env or default 5000
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
