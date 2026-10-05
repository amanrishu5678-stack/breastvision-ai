"""API tests. Run from backend/:  python -m unittest discover -s tests -v

The success-path test uses a TEST DOUBLE for the model (it only checks the HTTP
plumbing and response shape). It is never used by the real application.
"""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from app import create_app
from services.model_service import ModelService


def make_image(fmt="PNG", size=(128, 128)):
    buf = io.BytesIO()
    Image.new("RGB", size, (90, 90, 90)).save(buf, format=fmt)
    buf.seek(0)
    return buf


class StubPredictor:  # test double only
    config = {"model_name": "TestModel", "model_version": "0"}

    def predict(self, image, explain=True):
        return {"class": "A", "class_index": 0, "probability": 0.6,
                "probabilities": {"A": 0.6, "B": 0.4}}


class NoModelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        svc = ModelService(Path(self.tmp.name))
        svc.load()
        self.client = create_app(svc).test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def test_health_reports_no_model(self):
        r = self.client.get("/api/health").get_json()
        self.assertEqual(r["status"], "ok")
        self.assertFalse(r["model_loaded"])
        self.assertIn("not been trained", r["model_message"])

    def test_info_and_metrics_404(self):
        for url in ("/api/model-info", "/api/model-metrics"):
            r = self.client.get(url)
            self.assertEqual(r.status_code, 404)
            self.assertFalse(r.get_json()["success"])

    def test_analyze_503_without_model(self):
        r = self.client.post("/api/analyze", data={"image": (make_image(), "a.png")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 503)
        self.assertEqual(r.get_json()["error"]["code"], "MODEL_UNAVAILABLE")


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        svc = ModelService(Path(self.tmp.name))
        svc.predictor = StubPredictor()  # test double
        self.client = create_app(svc).test_client()

    def tearDown(self):
        self.tmp.cleanup()

    def post(self, data):
        return self.client.post("/api/analyze", data=data, content_type="multipart/form-data")

    def test_success_shape(self):
        r = self.post({"image": (make_image(), "scan.png", "image/png")})
        self.assertEqual(r.status_code, 200)
        j = r.get_json()
        self.assertTrue(j["success"])
        self.assertEqual(j["prediction"]["class"], "A")
        self.assertIn("analyzed_at", j)

    def test_jpeg_ok_and_predict_alias(self):
        r = self.client.post("/api/predict",
                             data={"image": (make_image("JPEG"), "s.jpg", "image/jpeg")},
                             content_type="multipart/form-data")
        self.assertEqual(r.status_code, 200)

    def test_no_file(self):
        self.assertEqual(self.post({}).get_json()["error"]["code"], "NO_FILE")

    def test_wrong_extension(self):
        r = self.post({"image": (io.BytesIO(b"hello"), "notes.txt", "text/plain")})
        self.assertEqual(r.status_code, 415)

    def test_fake_png_extension(self):
        r = self.post({"image": (io.BytesIO(b"not an image"), "x.png", "image/png")})
        self.assertEqual(r.status_code, 422)
        self.assertEqual(r.get_json()["error"]["code"], "UNREADABLE_IMAGE")

    def test_gif_renamed_to_png_rejected(self):
        buf = io.BytesIO()
        Image.new("RGB", (100, 100)).save(buf, format="GIF")
        buf.seek(0)
        r = self.post({"image": (buf, "x.png", "image/png")})
        self.assertEqual(r.status_code, 415)

    def test_truncated_image(self):
        data = make_image().getvalue()[:60]
        r = self.post({"image": (io.BytesIO(data), "x.png", "image/png")})
        self.assertEqual(r.status_code, 422)

    def test_too_small(self):
        r = self.post({"image": (make_image(size=(20, 20)), "x.png", "image/png")})
        self.assertEqual(r.get_json()["error"]["code"], "IMAGE_TOO_SMALL")

    def test_too_large(self):
        import config
        old = config.MAX_UPLOAD_MB
        config.MAX_UPLOAD_MB = 0  # any non-empty file exceeds the limit
        try:
            r = self.post({"image": (make_image(), "x.png", "image/png")})
            self.assertIn(r.status_code, (413,))
        finally:
            config.MAX_UPLOAD_MB = old

    def test_cors_allows_dev_origin_only(self):
        ok = self.client.get("/api/health", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(ok.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")
        bad = self.client.get("/api/health", headers={"Origin": "http://evil.example"})
        self.assertIsNone(bad.headers.get("Access-Control-Allow-Origin"))
        pre = self.client.options("/api/analyze", headers={"Origin": "http://localhost:5173"})
        self.assertIn(pre.status_code, (200, 204))
        self.assertEqual(pre.headers.get("Access-Control-Allow-Origin"), "http://localhost:5173")

    def test_unknown_route_json(self):
        r = self.client.get("/api/nope")
        self.assertEqual(r.status_code, 404)
        self.assertFalse(r.get_json()["success"])


class ArtifactEndpointTests(unittest.TestCase):
    """Checks that endpoints return whatever JSON is on disk (temp files, test only)."""

    def test_metrics_passthrough(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "config.json").write_text(json.dumps({"model_name": "M", "class_names": ["A"]}))
            (d / "metrics.json").write_text(json.dumps({"accuracy": 0.5, "split": "test"}))
            svc = ModelService(d)
            client = create_app(svc).test_client()
            m = client.get("/api/model-metrics").get_json()
            self.assertEqual(m["metrics"]["split"], "test")
            self.assertEqual(m["history"], [])
            i = client.get("/api/model-info").get_json()
            self.assertEqual(i["model"]["model_name"], "M")
            self.assertFalse(i["loaded"])


if __name__ == "__main__":
    unittest.main()
