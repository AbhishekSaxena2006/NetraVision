"""
Unit tests for NetraVision Flask Backend
Tests /api/health and /api/predict endpoints, validation, and error cases.
"""

import io
import os
import unittest
import json
from app import app, ALLOWED_EXTENSIONS

class NetraVisionAPITestCase(unittest.TestCase):
    def setUp(self):
        self.app = app
        self.client = self.app.test_client()

    def test_health_endpoint(self):
        """Test GET /api/health returns 200 and expected health metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data.get("status"), "healthy")
        self.assertTrue(data.get("model_loaded"))
        self.assertEqual(data.get("num_classes"), 5)
        self.assertIn("class_mapping", data)
        self.assertIn("disclaimer", data)
        print("\n[PASS] Health check verified.")

    def test_predict_no_file(self):
        """Test POST /api/predict without an image file returns 400 error."""
        response = self.client.post("/api/predict", data={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertIn("No image file provided", data.get("error"))
        print("[PASS] Missing file rejection verified.")

    def test_predict_invalid_extension(self):
        """Test POST /api/predict with disallowed file extension returns 400."""
        data = {
            "image": (io.BytesIO(b"fake text content"), "report.txt")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertEqual(json_data.get("status"), "error")
        self.assertIn("Invalid file type", json_data.get("error"))
        print("[PASS] Invalid extension rejection verified.")

    def test_predict_corrupt_image(self):
        """Test POST /api/predict with corrupt image data returns 400."""
        data = {
            "image": (io.BytesIO(b"corrupt binary data that is not an image"), "corrupt.png")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 400)
        json_data = response.get_json()
        self.assertEqual(json_data.get("status"), "error")
        print("[PASS] Corrupt image rejection verified.")

    def test_predict_valid_image_field_image(self):
        """Test POST /api/predict with a valid retinal fundus image using 'image' field."""
        sample_img_path = r"D:\aptos2019-blindness-detection\test_images\0005cfc8afb6.png"
        self.assertTrue(os.path.exists(sample_img_path), f"Sample image {sample_img_path} not found.")

        with open(sample_img_path, "rb") as f:
            file_bytes = f.read()

        data = {
            "image": (io.BytesIO(file_bytes), "0005cfc8afb6.png")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res = response.get_json()
        self.assertEqual(res.get("status"), "success")
        self.assertIn("predicted_class", res)
        self.assertIn("grade_name", res)
        self.assertIn("confidence", res)
        self.assertIn("probabilities", res)
        self.assertEqual(len(res["probabilities"]), 5)
        self.assertIn("recommendation", res)
        self.assertIn("disclaimer", res)

        # Probabilities should sum approximately to 1.0
        prob_sum = sum(res["probabilities"])
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        print(f"[PASS] Valid image prediction: Grade '{res['grade_name']}', Confidence: {res['confidence']}%")

    def test_predict_valid_image_field_file(self):
        """Test POST /api/predict with 'file' field name (alternate form key)."""
        sample_img_path = r"D:\aptos2019-blindness-detection\test_images\0005cfc8afb6.png"
        with open(sample_img_path, "rb") as f:
            file_bytes = f.read()

        data = {
            "file": (io.BytesIO(file_bytes), "sample.png")
        }
        response = self.client.post(
            "/api/predict",
            data=data,
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        res = response.get_json()
        self.assertEqual(res.get("status"), "success")
        self.assertEqual(res.get("predicted_class"), 1)
        print("[PASS] Alternate 'file' form key verified.")

    def test_serve_frontend(self):
        """Test static frontend files are accessible through Flask."""
        res_index = self.client.get("/")
        self.assertEqual(res_index.status_code, 200)
        self.assertIn(b"NetraVision", res_index.data)

        res_predict = self.client.get("/predict.html")
        self.assertEqual(res_predict.status_code, 200)
        self.assertIn(b"NetraVision", res_predict.data)
        print("[PASS] Static frontend serving verified.")

if __name__ == "__main__":
    unittest.main()

