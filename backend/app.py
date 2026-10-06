"""Flask REST API.

Run (from the backend/ folder):  python app.py
"""
import logging
from datetime import datetime, timezone

from flask import Flask, jsonify, request

import config
from services.model_service import TRAIN_HINT, ModelService
from utils.errors import ApiError, error_response
from utils.validation import validate_and_open_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def create_app(model_service: ModelService | None = None) -> Flask:
    app = Flask(__name__)
    # Request-size guard (extra 1 MB allows for multipart overhead)
    app.config["MAX_CONTENT_LENGTH"] = (config.MAX_UPLOAD_MB + 1) * 1024 * 1024

    @app.after_request
    def add_cors_headers(resp):
        """Allow the React dev server (and only configured origins) to call the API."""
        origin = request.headers.get("Origin")
        if origin and origin in config.CORS_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Vary"] = "Origin"
            resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return resp

    if model_service is None:
        model_service = ModelService()
        model_service.load()  # once, at startup - never retrains
    service = model_service

    @app.get("/api/health")
    def health():
        return jsonify({
            "status": "ok",
            "model_loaded": service.loaded,
            "model_message": None if service.loaded else service.load_error,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    @app.get("/api/model-info")
    def model_info():
        info = service.model_info()
        if info is None:
            raise ApiError("MODEL_NOT_TRAINED", TRAIN_HINT, 404)
        return jsonify({"success": True, **info})

    @app.get("/api/model-metrics")
    def model_metrics():
        data = service.model_metrics()
        if data is None:
            raise ApiError(
                "METRICS_NOT_AVAILABLE",
                "No evaluation metrics found. From the ml/ folder run: python evaluate.py",
                404,
            )
        return jsonify({"success": True, **data})

    @app.post("/api/analyze")
    @app.post("/api/predict")
    def analyze():
        if not service.loaded:
            raise ApiError("MODEL_UNAVAILABLE", service.load_error or TRAIN_HINT, 503)
        image = validate_and_open_image(request.files.get("image"))
        explain = request.form.get("explain", "true").lower() != "false"
        try:
            return jsonify(service.analyze(image, explain=explain))
        except Exception:  # noqa: BLE001
            app.logger.exception("Inference failed")
            raise ApiError("INFERENCE_FAILED", "The model could not analyze this image.", 500)

    @app.errorhandler(ApiError)
    def handle_api_error(err: ApiError):
        return error_response(err.code, err.message, err.status)

    @app.errorhandler(413)
    def too_large(_):
        return error_response("FILE_TOO_LARGE", f"The image is larger than {config.MAX_UPLOAD_MB} MB.", 413)

    @app.errorhandler(404)
    def not_found(_):
        return error_response("NOT_FOUND", "Endpoint not found.", 404)

    @app.errorhandler(405)
    def bad_method(_):
        return error_response("METHOD_NOT_ALLOWED", "Method not allowed for this endpoint.", 405)

    @app.errorhandler(500)
    def server_error(_):
        return error_response("SERVER_ERROR", "Unexpected server error.", 500)

    return app


app = create_app()

if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=config.PORT, debug=config.DEBUG)

