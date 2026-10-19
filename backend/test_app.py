import io
import json
from pathlib import Path

import pytest
from PIL import Image

# Import the Flask app
from app import app, MOCK_MODE

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "ok"
    assert isinstance(data["mock_mode"], bool)

def test_predict_no_file(client):
    resp = client.post("/api/predict", data={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()

def test_predict_invalid_extension(client):
    data = {
        "image": (io.BytesIO(b"not-an-image"), "test.txt"),
    }
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 400
    json_data = resp.get_json()
    assert "error" in json_data
    assert "allowed" in json_data["error"].lower()

def generate_png_bytes():
    img = Image.new("RGB", (100, 100), color="red")
    byte_io = io.BytesIO()
    img.save(byte_io, format="PNG")
    byte_io.seek(0)
    return byte_io

def test_predict_mock_success(client):
    # Ensure we are in mock mode (model file missing)
    assert MOCK_MODE is True
    png_bytes = generate_png_bytes()
    data = {"image": (png_bytes, "sample.png")}
    resp = client.post("/api/predict", data=data, content_type="multipart/form-data")
    assert resp.status_code == 200
    json_data = resp.get_json()
    # Verify required fields
    assert "predicted_class" in json_data
    assert "class_name" in json_data
    assert "confidence" in json_data
    assert "scores" in json_data
    assert json_data["mock_mode"] is True
    # Scores should contain 5 entries (0-4)
    assert set(json_data["scores"].keys()) == {"0", "1", "2", "3", "4"}
    # Confidence should be a float between 0 and 1
    assert 0.0 <= json_data["confidence"] <= 1.0
