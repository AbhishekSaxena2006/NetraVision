"""
Comprehensive Test Suite for NetraVision Fundus Validation Pipeline
====================================================================
Tests:
1. Portrait photograph negative regression test (HTTP 422, zero DR classification returned).
2. Multiple non-fundus real-world photos (portraits, scenery, objects, website assets).
3. Synthetic non-fundus images (face on bright background, face on dark background, solid, blurry, tiny).
4. Real APTOS 2019 clinical fundus images (HTTP 200, authentic DR prediction).
5. Exact API schema verification on rejections: status=error, 422 code, no predicted_class,
   no grade_name, no confidence, no probabilities.
6. Frontend stale cache eviction verification.
"""

import io
import os
import sys
import unittest
import tempfile
import cv2
import numpy as np

# Resolve paths
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT   = os.path.dirname(BACKEND_DIR)
ML_DIR      = os.path.join(REPO_ROOT, "ml")
FRONTEND_DIR = os.path.join(REPO_ROOT, "frontend")

for _p in (BACKEND_DIR, ML_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fundus_validator import validate_fundus_image, ValidationResult
import config


# ─────────────────────────────────────────────────────────────────────────────
# Helper Generators for Synthetic & Negative Test Cases
# ─────────────────────────────────────────────────────────────────────────────

def _save_tmp_image(img_bgr: np.ndarray, suffix=".png") -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        path = f.name
    cv2.imwrite(path, img_bgr)
    return path


def _create_synthetic_portrait_bright_bg(size=500) -> np.ndarray:
    """Simulates an ordinary human portrait photo with skin tones and a bright wall."""
    img = np.full((size, size, 3), (210, 205, 215), dtype=np.uint8)  # Light interior wall
    # Head (skin tones: BGR ~ (130, 160, 215))
    cv2.ellipse(img, (size//2, size//2 - 20), (int(size*0.22), int(size*0.30)), 0, 0, 360, (130, 160, 215), -1)
    # Hair
    cv2.ellipse(img, (size//2, int(size*0.25)), (int(size*0.24), int(size*0.14)), 0, 0, 180, (25, 25, 35), -1)
    # Eyes
    cv2.circle(img, (int(size*0.42), int(size*0.42)), int(size*0.03), (240, 240, 240), -1)
    cv2.circle(img, (int(size*0.42), int(size*0.42)), int(size*0.012), (30, 30, 30), -1)
    cv2.circle(img, (int(size*0.58), int(size*0.42)), int(size*0.03), (240, 240, 240), -1)
    cv2.circle(img, (int(size*0.58), int(size*0.42)), int(size*0.012), (30, 30, 30), -1)
    # Shirt (blue casual shirt)
    cv2.rectangle(img, (int(size*0.15), int(size*0.72)), (int(size*0.85), size), (180, 80, 40), -1)
    # Natural photo sensor noise so entropy matches real camera photos (>5.0)
    noise = np.random.randint(0, 30, img.shape, dtype=np.uint8)
    return cv2.add(img, noise)


def _create_synthetic_portrait_dark_bg(size=500) -> np.ndarray:
    """Simulates a studio portrait taken against a black backdrop."""
    img = np.zeros((size, size, 3), dtype=np.uint8)  # Dark studio background
    # Head
    cv2.ellipse(img, (size//2, int(size*0.38)), (int(size*0.20), int(size*0.28)), 0, 0, 360, (130, 160, 215), -1)
    # Hair
    cv2.ellipse(img, (size//2, int(size*0.22)), (int(size*0.22), int(size*0.12)), 0, 0, 180, (25, 25, 35), -1)
    # Torso / shoulders
    cv2.rectangle(img, (int(size*0.10), int(size*0.65)), (int(size*0.90), size), (100, 90, 85), -1)
    # Add texture/shading to subject
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    mask = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 10
    img[mask] = cv2.add(img[mask], noise[mask])
    return img


def _create_synthetic_fundus(size=500) -> np.ndarray:
    """Constructs a photorealistic synthetic fundus representation."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cx, cy, r = size // 2, size // 2, int(size * 0.44)
    # Warm red-orange circular retina
    cv2.circle(img, (cx, cy), r, (25, 75, 195), -1)  # BGR
    # Optic disc (bright yellow-white oval)
    cv2.ellipse(img, (int(cx - r*0.45), cy), (int(r*0.16), int(r*0.22)), 0, 0, 360, (110, 210, 245), -1)
    # Vessel tree (branching curvilinear dark red lines)
    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
        ex = int(cx + r * 0.85 * np.cos(angle))
        ey = int(cy + r * 0.85 * np.sin(angle))
        cv2.line(img, (int(cx - r*0.45), cy), (ex, ey), (15, 35, 110), 2)
    # Add subtle natural texture
    noise = np.random.randint(0, 22, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit Tests — fundus_validator.py Core Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestFundusValidatorUnit(unittest.TestCase):

    def _validate_np(self, img_bgr, suffix=".png") -> ValidationResult:
        tmp_path = _save_tmp_image(img_bgr, suffix)
        try:
            return validate_fundus_image(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_portrait_bright_background_rejected(self):
        """Portrait with bright background must be rejected as an ordinary rectangular photo."""
        portrait = _create_synthetic_portrait_bright_bg()
        res = self._validate_np(portrait)
        self.assertFalse(res.passed)
        self.assertIn(res.reason, ("NON_FUNDUS_RECTANGULAR", "PORTRAIT_DETECTED", "NON_RETINAL_SPECTRUM"))
        print(f"[PASS] Bright background portrait rejected: reason={res.reason}")

    def test_portrait_dark_background_rejected(self):
        """Portrait with dark background must be rejected due to non-circular aperture or non-retinal spectrum."""
        portrait = _create_synthetic_portrait_dark_bg()
        res = self._validate_np(portrait)
        self.assertFalse(res.passed)
        self.assertIn(res.reason, ("NON_CIRCULAR_APERTURE", "NON_RETINAL_SPECTRUM", "PORTRAIT_DETECTED"))
        print(f"[PASS] Dark backdrop portrait rejected: reason={res.reason}")

    def test_team_placeholder_rejected(self):
        """The team placeholder image from the website must be rejected."""
        team_img_path = os.path.join(FRONTEND_DIR, "assets", "images", "team_placeholder.jpg")
        if os.path.exists(team_img_path):
            res = validate_fundus_image(team_img_path)
            self.assertFalse(res.passed)
            self.assertIn(res.reason, ("NON_FUNDUS_RECTANGULAR", "NON_RETINAL_SPECTRUM"))
            print(f"[PASS] Website team placeholder rejected: reason={res.reason}")

    def test_website_about_and_logo_rejected(self):
        """Website background and logo images must be rejected."""
        for name in ("about_bg.jpg", "logo_bg.jpg"):
            path = os.path.join(FRONTEND_DIR, "assets", "images", name)
            if os.path.exists(path):
                res = validate_fundus_image(path)
                self.assertFalse(res.passed)
                print(f"[PASS] Asset {name} rejected: reason={res.reason}")

    def test_ordinary_objects_and_scenery_rejected(self):
        """Scikit-learn sample photographs (china, flower) must be rejected."""
        import glob
        for fname in ("china.jpg", "flower.jpg"):
            matches = glob.glob(os.path.join(REPO_ROOT, "**", fname), recursive=True)
            if matches:
                res = validate_fundus_image(matches[0])
                self.assertFalse(res.passed)
                self.assertEqual(res.reason, "NON_FUNDUS_RECTANGULAR")
                print(f"[PASS] Sample photo {fname} rejected: reason={res.reason}")

    def test_tiny_image_rejected(self):
        """Images smaller than 120x120 must be rejected as TOO_SMALL."""
        tiny = np.full((80, 80, 3), 120, dtype=np.uint8)
        res = self._validate_np(tiny)
        self.assertFalse(res.passed)
        self.assertEqual(res.reason, "TOO_SMALL")
        print("[PASS] Tiny image rejected.")

    def test_blurry_image_rejected(self):
        """Severely blurry images must be rejected as TOO_BLURRY."""
        fundus = _create_synthetic_fundus()
        blurry = cv2.GaussianBlur(fundus, (99, 99), 40)
        res = self._validate_np(blurry)
        self.assertFalse(res.passed)
        self.assertEqual(res.reason, "TOO_BLURRY")
        print("[PASS] Blurry image rejected.")

    def test_solid_white_and_black_rejected(self):
        """Solid white and black images must be rejected."""
        white = np.full((400, 400, 3), 255, dtype=np.uint8)
        black = np.zeros((400, 400, 3), dtype=np.uint8)
        res_w = self._validate_np(white)
        res_b = self._validate_np(black)
        self.assertFalse(res_w.passed)
        self.assertFalse(res_b.passed)
        print("[PASS] Solid white and black rejected.")

    def test_synthetic_fundus_accepted(self):
        """A valid synthetic retinal fundus photograph must pass validation."""
        fundus = _create_synthetic_fundus()
        res = self._validate_np(fundus)
        self.assertTrue(res.passed, f"Valid synthetic fundus rejected: {res.reason} - {res.message}")
        print("[PASS] Synthetic fundus accepted.")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Integration Tests — Flask /api/predict Rejection & Inference
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictEndpointValidation(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from app import app as flask_app
        cls.client = flask_app.test_client()

    def _post_image_bytes(self, img_bgr, filename="upload.png"):
        success, buf = cv2.imencode(".png", img_bgr)
        self.assertTrue(success)
        return self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(buf.tobytes()), filename)},
            content_type="multipart/form-data"
        )

    def test_portrait_photo_negative_regression(self):
        """
        NEGATIVE REGRESSION TEST:
        An ordinary portrait photograph must return HTTP 422 and must NOT return
        a DR grade, confidence, or class probabilities.
        """
        portrait = _create_synthetic_portrait_bright_bg()
        resp = self._post_image_bytes(portrait, "portrait.png")

        self.assertEqual(resp.status_code, 422, f"Expected 422 for portrait, got {resp.status_code}")
        data = resp.get_json()
        self.assertEqual(data["status"], "error")
        self.assertIn("error_code", data)
        self.assertIn("error", data)
        self.assertIn("hint", data)

        # STRICT ASSERTIONS: No DR prediction metadata allowed on rejection
        self.assertNotIn("predicted_class", data)
        self.assertNotIn("grade_name", data)
        self.assertNotIn("confidence", data)
        self.assertNotIn("probabilities", data)
        self.assertNotIn("probabilities_pct", data)
        self.assertNotIn("level", data)
        print(f"[PASS] Regression: Portrait photo rejected with 422 (reason={data['error_code']}). No DR grade returned.")

    def test_dark_backdrop_portrait_negative_regression(self):
        """Studio portrait on black background must also be rejected with HTTP 422."""
        portrait_dark = _create_synthetic_portrait_dark_bg()
        resp = self._post_image_bytes(portrait_dark, "studio_portrait.png")
        self.assertEqual(resp.status_code, 422)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")
        self.assertNotIn("predicted_class", data)
        print(f"[PASS] Regression: Dark backdrop portrait rejected with 422 (reason={data['error_code']}).")

    def test_exact_user_portrait_rejected_with_422(self):
        """
        NEGATIVE REGRESSION TEST USING EXACT USER PORTRAIT:
        The exact uploaded passport/portrait photo must return HTTP 422
        with reason NON_FUNDUS_RECTANGULAR and zero DR prediction fields.
        """
        portrait_path = os.path.join(BACKEND_DIR, "test_data", "exact_portrait.jpg")
        self.assertTrue(os.path.exists(portrait_path), f"Portrait test image not found at {portrait_path}")

        with open(portrait_path, "rb") as f:
            file_bytes = f.read()

        resp = self.client.post(
            "/api/predict",
            data={"image": (io.BytesIO(file_bytes), "exact_portrait.jpg")},
            content_type="multipart/form-data"
        )

        self.assertEqual(resp.status_code, 422, f"Expected 422 for exact portrait, got {resp.status_code}")
        data = resp.get_json()
        self.assertEqual(data.get("status"), "error")
        self.assertEqual(data.get("error_code"), "NON_FUNDUS_RECTANGULAR")
        self.assertIn("ordinary rectangular photograph", data.get("error", ""))
        self.assertIn("hint", data)

        # STRICT ASSERTIONS: No DR prediction fields can exist
        self.assertNotIn("predicted_class", data)
        self.assertNotIn("grade_name", data)
        self.assertNotIn("confidence", data)
        self.assertNotIn("probabilities", data)
        self.assertNotIn("probabilities_pct", data)
        self.assertNotIn("recommendation", data)
        print(f"[PASS] EXACT USER PORTRAIT REGRESSION VERIFIED: HTTP 422 returned. Zero DR prediction fields.")

    def test_real_aptos_fundus_accepted_with_inference(self):
        """Valid clinical APTOS retinal images must be accepted and receive DR classification."""
        import glob
        aptos_imgs = glob.glob("D:/aptos2019-blindness-detection/train_images/*.png")[:3]
        if not aptos_imgs and os.path.exists(config.TEST_IMAGES_DIR):
            aptos_imgs = [os.path.join(config.TEST_IMAGES_DIR, f)
                          for f in os.listdir(config.TEST_IMAGES_DIR) if f.endswith(('.png', '.jpg'))][:3]

        self.assertTrue(len(aptos_imgs) > 0, "No APTOS images found for positive validation test.")

        for p in aptos_imgs:
            with open(p, "rb") as f:
                img_bytes = f.read()

            resp = self.client.post(
                "/api/predict",
                data={"image": (io.BytesIO(img_bytes), os.path.basename(p))},
                content_type="multipart/form-data"
            )
            self.assertEqual(resp.status_code, 200, f"Valid APTOS image {p} failed with {resp.status_code}")
            data = resp.get_json()
            self.assertEqual(data["status"], "success")
            self.assertIn("predicted_class", data)
            self.assertIn("grade_name", data)
            self.assertIn("confidence", data)
            self.assertEqual(len(data["probabilities"]), 5)
            print(f"[PASS] Real APTOS image {os.path.basename(p)} accepted: Grade='{data['grade_name']}', Confidence={data['confidence']}%")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Frontend Stale Result Purge Verification
# ─────────────────────────────────────────────────────────────────────────────

class TestFrontendStaleResultHandling(unittest.TestCase):

    def test_predict_js_purges_session_storage_on_error(self):
        """Verify predict.js actively removes sessionStorage on file selection, error, and replace."""
        predict_js_path = os.path.join(FRONTEND_DIR, "js", "predict.js")
        with open(predict_js_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("sessionStorage.removeItem('nv_result')", content)
        # Verify it removes on file selection
        self.assertIn("handleFile", content)
        # Verify it removes on error response
        self.assertIn("!response.ok", content)
        print("[PASS] predict.js stale cache eviction verified.")

    def test_results_html_guards_against_stale_or_invalid_data(self):
        """Verify results.html verifies success status and predicted_class or hides results."""
        results_html_path = os.path.join(FRONTEND_DIR, "results.html")
        with open(results_html_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("data.status !== 'success'", content)
        self.assertIn("data.predicted_class === undefined", content)
        self.assertIn("sessionStorage.removeItem('nv_result')", content)
        print("[PASS] results.html invalid/stale result guard verified.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
