"""
NetraVision — Flask Backend for Diabetic Retinopathy Detection

EDUCATIONAL & RESEARCH PROTOTYPE DISCLAIMER:
--------------------------------------------
NetraVision is an educational research prototype developed for academic demonstration.
It is NOT clinically validated, NOT certified as a medical device, and MUST NOT be used
as a substitute for professional medical advice, clinical diagnosis, or treatment decisions.
Always consult a qualified ophthalmologist or healthcare professional for retinal evaluation.
"""

import os
import sys
import tempfile
import logging
from typing import Tuple, Dict, Any

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import torch
import torch.nn.functional as F
import cv2

# Fundus image heuristic validator (must import after sys.path is configured below)
# Deferred import happens after ML path setup — see bottom of startup block.

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("NetraVisionBackend")

# Resolve repository paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))            # NetraVision/backend
REPO_ROOT = os.path.dirname(BACKEND_DIR)                            # NetraVision
ML_DIR = os.path.join(REPO_ROOT, "ml")                              # NetraVision/ml
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")                  # NetraVision/frontend
OUTPUTS_DIR = os.path.join(REPO_ROOT, "outputs")                    # NetraVision/outputs

# Add ML directory and Backend directory to Python path
if os.path.exists(ML_DIR) and ML_DIR not in sys.path:
    sys.path.insert(0, ML_DIR)
    logger.info("Added ML module directory to sys.path: %s", ML_DIR)

if os.path.exists(BACKEND_DIR) and BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)
    logger.info("Added Backend directory to sys.path: %s", BACKEND_DIR)

# Import model configuration, preprocessing, and mappings from ml/
try:
    import config
    from predict import load_inference_model, preprocess_image, CLINICAL_RECOMMENDATIONS
    from model import get_model
    logger.info("Successfully imported modules from NetraVision ML package.")
except ImportError as e:
    logger.error("Failed to import ML modules from %s: %s", ML_DIR, e)
    raise RuntimeError(
        f"Unable to load ML dependencies from '{ML_DIR}'. "
        "Please ensure the directory contains config.py, predict.py, and model.py."
    ) from e

# Import fundus image heuristic validator (fail closed — never run without validator)
try:
    try:
        from fundus_validator import validate_fundus_image
    except ImportError:
        from backend.fundus_validator import validate_fundus_image
    logger.info("Fundus image validator loaded successfully.")
except ImportError as e:
    logger.critical("FATAL: Could not import fundus_validator: %s", e)
    raise RuntimeError(
        f"FATAL: fundus_validator could not be loaded from '{BACKEND_DIR}'. "
        "Strict retinal image validation is mandatory."
    ) from e

# Model checkpoint configuration (defaults to NetraVision/outputs/best_model.pth)
DEFAULT_MODEL_PATH = os.path.join(OUTPUTS_DIR, "best_model.pth")
MODEL_PATH = os.environ.get("NETRAVISION_MODEL_PATH", DEFAULT_MODEL_PATH)
DEVICE = config.DEVICE

# Initialize Flask app with frontend static directory
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# Configuration
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
VERSION = "2.3.0-strict-fundus-guard"
DISCLAIMER_TEXT = (
    "NetraVision is an educational and academic research prototype. "
    "It is NOT clinically validated, approved as a diagnostic tool, or a substitute "
    "for professional clinical assessment by a certified ophthalmologist."
)

@app.after_request
def add_cache_prevention_headers(response):
    """Disable aggressive browser caching so frontend JS updates take effect immediately."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response

# Load PyTorch model at startup (with lazy-load fallback)
def get_model_instance():
    global model
    if model is None:
        try:
            logger.info("Loading PyTorch model from: %s on device: %s", MODEL_PATH, DEVICE)
            model = load_inference_model(weights_path=MODEL_PATH, device=DEVICE)
            model.eval()
            logger.info("Model loaded successfully and set to evaluation mode.")
        except Exception as e:
            logger.error("Error loading model checkpoint: %s", e)
            model = None
    return model

model = None
try:
    model = get_model_instance()
except Exception as e:
    logger.warning("Startup model load deferred: %s", e)

def allowed_file(filename: str) -> bool:
    """Checks whether the filename has an allowed extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "status": "error",
        "error": "File size exceeds the 16MB maximum upload limit."
    }), 413

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        "status": "error",
        "error": "Method not allowed for this endpoint."
    }), 405

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "status": "error",
        "error": "Endpoint or resource not found."
    }), 404

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "status": "error",
        "error": "An internal server error occurred."
    }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify backend operational state and model availability.
    """
    is_loaded = model is not None
    response = {
        "status": "healthy" if is_loaded else "degraded",
        "backend": "NetraVision Flask Backend",
        "version": VERSION,
        "validator_active": validate_fundus_image is not None,
        "model_loaded": is_loaded,
        "model_path": MODEL_PATH,
        "device": str(DEVICE),
        "num_classes": config.NUM_CLASSES,
        "class_mapping": config.CLASS_NAMES,
        "allowed_extensions": list(ALLOWED_EXTENSIONS),
        "disclaimer": DISCLAIMER_TEXT
    }
    status_code = 200 if is_loaded else 503
    return jsonify(response), status_code

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Predict Diabetic Retinopathy grade for an uploaded retinal fundus image.
    Accepts multipart/form-data with file field 'image' or 'file'.
    Reuses exact preprocessing and class mapping from predict.py.
    """
    if model is None:
        return jsonify({
            "status": "error",
            "error": "Model weights are not loaded. Server is currently unavailable for predictions."
        }), 503

    # Check request files
    file = request.files.get("image") or request.files.get("file")
    if file is None and len(request.files) > 0:
        file = next(iter(request.files.values()))

    if file is None:
        return jsonify({
            "status": "error",
            "error": "No image file provided. Please submit an image under key 'image' or 'file'."
        }), 400

    filename = file.filename
    if not filename or filename.strip() == "":
        return jsonify({
            "status": "error",
            "error": "No file selected. Please select a valid image file."
        }), 400

    if not allowed_file(filename):
        return jsonify({
            "status": "error",
            "error": f"Invalid file type. Allowed image formats are: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
        }), 400

    # Save uploaded file to a temporary file for exact preprocessing
    temp_file_path = None
    try:
        ext = os.path.splitext(secure_filename(filename))[1]
        if not ext:
            ext = ".png"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        file_size_bytes = os.path.getsize(temp_file_path)
        if file_size_bytes == 0:
            return jsonify({
                "status": "error",
                "error": "Uploaded image file is empty (0 bytes)."
            }), 400

        logger.info("=" * 60)
        logger.info("[NETRAVISION PREDICT] Incoming request: file='%s' size=%d bytes", filename, file_size_bytes)

        # ── Dedicated Fundus-vs-Non-Fundus Validation Step ───────────────────
        # Runs BEFORE deep learning preprocessing and inference.
        # Enforces optical aperture, circular disc geometry, scale-invariant
        # sharpness, and physiological retinal chromaticity.
        if validate_fundus_image is None:
            logger.critical("[FATAL] Fundus validator is not loaded. Refusing unverified prediction.")
            return jsonify({
                "status": "error",
                "error": "Image validation service is unavailable. Refusing unverified prediction.",
                "hint": "Please restart backend server."
            }), 500

        logger.info("[VALIDATOR] Running multi-domain fundus validation on '%s'...", filename)
        val_result = validate_fundus_image(temp_file_path)
        if not val_result.passed:
            logger.warning(
                "[VALIDATOR REJECTED] file='%s' | error_code='%s' | reason='%s'",
                filename, val_result.reason, val_result.message
            )
            logger.warning("[VALIDATOR DETAILS] %s", val_result.details)
            # 400 for corrupted/unreadable image bytes; 422 for non-fundus semantic validation rejections
            status_code = 400 if val_result.reason == "UNREADABLE" else 422
            return jsonify({
                "status": "error",
                "error_code": val_result.reason,
                "error": val_result.message,
                "validation_details": val_result.details,
                "hint": getattr(val_result, "hint", "Please upload a clear retinal fundus photograph.")
            }), status_code

        logger.info("[VALIDATOR PASSED] file='%s' verified as authentic fundus image. Proceeding to DR model inference.", filename)
        for w in getattr(val_result, "warnings", []):
            logger.warning("[VALIDATOR WARNING] %s: %s", filename, w)
        # ────────────────────────────────────────────────────────────────────

        # Reuse exact preprocessing from predict.py
        # preprocess_image reads file with cv2.imread and applies Ben Graham preprocessing & normalization
        try:
            tensor, original_img, processed_img = preprocess_image(temp_file_path)
        except ValueError as ve:
            return jsonify({
                "status": "error",
                "error": f"Corrupt or unreadable image file: {str(ve)}"
            }), 400
        except Exception as pe:
            logger.error("Preprocessing error: %s", pe, exc_info=True)
            return jsonify({
                "status": "error",
                "error": f"Preprocessing failed: {str(pe)}"
            }), 400

        # Run model inference
        tensor = tensor.to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

        # Clinical recommendation and class mapping from existing training & predict logic
        grade_name = config.CLASS_NAMES[predicted_class]
        rec = CLINICAL_RECOMMENDATIONS.get(predicted_class, {
            "status": f"Grade {predicted_class}",
            "severity": "Unknown",
            "action": "Consult an ophthalmologist for evaluation."
        })

        probabilities_list = [round(float(p), 4) for p in probs]
        probabilities_pct = [round(float(p) * 100, 2) for p in probs]
        probabilities_dict = {
            config.CLASS_NAMES[i]: round(float(probs[i]), 4)
            for i in range(config.NUM_CLASSES)
        }

        response_payload = {
            "status": "success",
            # Core required fields
            "predicted_class": predicted_class,
            "grade_name": grade_name,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities_list,
            # Frontend compatibility fields (matching predict.js and results.html)
            "level": predicted_class,
            "label": f"{grade_name} DR" if predicted_class > 0 else "No DR (Normal)",
            "confidence_score": round(confidence, 4),
            "probabilities_pct": probabilities_pct,
            "probabilities_by_class": probabilities_dict,
            "recommendation": rec,
            "filename": filename,
            # Educational prototype notice
            "disclaimer": DISCLAIMER_TEXT
        }

        return jsonify(response_payload), 200

    except Exception as e:
        logger.error("Unexpected prediction error: %s", e, exc_info=True)
        return jsonify({
            "status": "error",
            "error": f"Internal prediction error: {str(e)}"
        }), 500

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception as ce:
                logger.warning("Could not delete temporary file %s: %s", temp_file_path, ce)

# Static file serving from frontend/
@app.route("/", defaults={"path": "index.html"})
@app.route("/<path:path>")
def serve_frontend(path):
    if path.startswith("api/"):
        return jsonify({"status": "error", "error": "Not found"}), 404
    file_path = os.path.join(FRONTEND_DIR, path)
    if os.path.isfile(file_path):
        return send_from_directory(FRONTEND_DIR, path)
    if os.path.isfile(file_path + ".html"):
        return send_from_directory(FRONTEND_DIR, path + ".html")
    return jsonify({"status": "error", "error": "Resource not found"}), 404

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    logger.info("=" * 65)
    logger.info("  NETRAVISION BACKEND SERVER - v%s", VERSION)
    logger.info("  Fundus Image Verification Engine: %s", "ACTIVE (STRICT)" if validate_fundus_image else "INACTIVE")
    logger.info("  Starting at http://%s:%d", host, port)
    logger.info("  Serving frontend from: %s", FRONTEND_DIR)
    logger.info("=" * 65)
    logger.info("DISCLAIMER: %s", DISCLAIMER_TEXT)
    app.run(host=host, port=port, debug=False)
