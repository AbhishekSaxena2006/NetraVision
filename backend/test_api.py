"""
Unit and Integration tests for NetraVision Flask Backend & Frontend Integration
Tests:
- /api/health and /api/predict endpoints
- Input validation and error handling
- Verification that frontend JS connects to real /api/predict
- Verification that results.html renders real model probabilities without mock formulas
"""

import io
import os
import unittest
import json
from app import app, ALLOWED_EXTENSIONS, FRONTEND_DIR
import config

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

    def _get_sample_image_path(self):
        sample_img_path = os.path.join(config.TEST_IMAGES_DIR, "0005cfc8afb6.png")
        if not os.path.exists(sample_img_path) and os.path.exists(config.TEST_IMAGES_DIR):
            files = [f for f in os.listdir(config.TEST_IMAGES_DIR) if f.endswith(('.png', '.jpg'))]
            if files:
                sample_img_path = os.path.join(config.TEST_IMAGES_DIR, files[0])
        return sample_img_path

    def test_predict_valid_image_field_image(self):
        """Test POST /api/predict with a valid retinal fundus image using 'image' field."""
        sample_img_path = self._get_sample_image_path()
        self.assertTrue(os.path.exists(sample_img_path), f"Sample image {sample_img_path} not found.")

        with open(sample_img_path, "rb") as f:
            file_bytes = f.read()

        data = {
            "image": (io.BytesIO(file_bytes), os.path.basename(sample_img_path))
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
        self.assertIn("probabilities_pct", res)
        self.assertEqual(len(res["probabilities_pct"]), 5)
        self.assertIn("recommendation", res)
        self.assertIn("disclaimer", res)

        # Probabilities should sum approximately to 1.0
        prob_sum = sum(res["probabilities"])
        self.assertAlmostEqual(prob_sum, 1.0, places=2)

        print(f"[PASS] Valid image prediction: Grade '{res['grade_name']}', Confidence: {res['confidence']}%")

    def test_predict_valid_image_field_file(self):
        """Test POST /api/predict with 'file' field name (alternate form key)."""
        sample_img_path = self._get_sample_image_path()
        with open(sample_img_path, "rb") as f:
            file_bytes = f.read()

        data = {
            "file": (io.BytesIO(file_bytes), os.path.basename(sample_img_path))
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

        res_results = self.client.get("/results.html")
        self.assertEqual(res_results.status_code, 200)
        self.assertIn(b"NetraVision", res_results.data)
        print("[PASS] Static frontend serving verified.")

    def test_frontend_predict_js_connected(self):
        """Verify predict.js calls the real /api/predict endpoint and no longer uses simulateDemoPrediction."""
        predict_js_path = os.path.join(FRONTEND_DIR, "js", "predict.js")
        with open(predict_js_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("/api/predict", content)
        self.assertIn("formData.append('image', selectedFile)", content)
        self.assertNotIn("simulateDemoPrediction", content)
        print("[PASS] predict.js verification passed: real API fetch confirmed, demo simulator removed.")

    def test_frontend_results_html_uses_real_probabilities(self):
        """Verify results.html consumes real probabilities and no longer uses generateProbabilities."""
        results_html_path = os.path.join(FRONTEND_DIR, "results.html")
        with open(results_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("data.probabilities", content)
        self.assertNotIn("generateProbabilities", content)
        self.assertNotIn("Demo Mode \u2014 Simulated Results", content)
        print("[PASS] results.html verification passed: real probabilities confirmed, demo formulas removed.")

if __name__ == "__main__":
    unittest.main()
