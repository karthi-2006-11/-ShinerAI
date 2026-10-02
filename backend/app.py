"""
ShinerAI: Flask REST Application Factory & Route Definitions
Serves prediction and explainability endpoints for low-DO early warning.
"""

from datetime import datetime, timezone
import os
from typing import Optional

from flask import Flask, jsonify, request

from backend.config import DECISION_THRESHOLD, MODEL_ARTIFACT_PATH
from backend.explainability import ExplainabilityService
from backend.model_service import ModelService
from backend.validation import ValidationError, validate_prediction_payload


def create_app(model_artifact_path: Optional[str] = None) -> Flask:
    """
    Application factory for the ShinerAI REST API.

    Parameters
    ----------
    model_artifact_path : str, optional
        Path to the joblib model artifact. If None, uses MODEL_ARTIFACT_PATH.

    Returns
    -------
    Flask
        Configured Flask application instance.
    """
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Instantiate model and explainability services
    model_service = ModelService(model_artifact_path)
    explainability_service = ExplainabilityService(model_service.model)

    # --------------------------------------------------------------------------
    # Error Handlers
    # --------------------------------------------------------------------------
    @app.errorhandler(ValidationError)
    def handle_validation_error(err: ValidationError):
        return jsonify(err.to_dict()), 400

    @app.errorhandler(400)
    def handle_bad_request(err):
        return jsonify({
            "error": "BadRequest",
            "message": getattr(err, "description", "Malformed or unparseable JSON payload."),
        }), 400

    @app.errorhandler(404)
    def handle_not_found(err):
        return jsonify({
            "error": "NotFound",
            "message": "The requested endpoint does not exist. Available: /, /health, /model-info, /predict, /explain.",
        }), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(err):
        return jsonify({
            "error": "MethodNotAllowed",
            "message": f"Method {request.method} is not permitted for this endpoint.",
        }), 405

    @app.errorhandler(500)
    def handle_internal_error(err):
        return jsonify({
            "error": "InternalServerError",
            "message": "An unexpected error occurred during request processing.",
        }), 500

    # --------------------------------------------------------------------------
    # API Routes
    # --------------------------------------------------------------------------
    @app.route("/", methods=["GET"])
    def index():
        return jsonify({
            "project": "ShinerAI",
            "title": "AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms",
            "status": "online",
            "phase": "Phase 4 - Model Serving & Explainability",
            "active_model": model_service.artifact_name,
            "endpoints": {
                "GET /health": "Server health, model load status, and runtime environment.",
                "GET /model-info": "Model architecture, feature schema, decision threshold, and held-out test metrics.",
                "POST /predict": "Predict whether pond DO will drop below 3.0 mg/L within 2 hours.",
                "POST /explain": "Generate prediction along with local SHAP feature attributions.",
            },
        }), 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "healthy",
            "project": "ShinerAI",
            "model_loaded": True,
            "model_artifact": model_service.artifact_name,
            "model_family": model_service.model_family,
            "decision_threshold": DECISION_THRESHOLD,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }), 200

    @app.route("/model-info", methods=["GET"])
    def model_info():
        return jsonify(model_service.get_model_info()), 200

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Missing or malformed JSON request body.")

        validated_data = validate_prediction_payload(payload)
        prediction_result = model_service.predict(validated_data)
        return jsonify(prediction_result), 200

    @app.route("/explain", methods=["POST"])
    def explain():
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError("Missing or malformed JSON request body.")

        validated_data = validate_prediction_payload(payload)
        prediction_result = model_service.predict(validated_data)
        explanation_result = explainability_service.explain(validated_data)

        # Merge prediction and local explanation
        response = {**prediction_result, **explanation_result}
        return jsonify(response), 200

    return app


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app = create_app()
    app.run(host="0.0.0.0", port=port, debug=False)
