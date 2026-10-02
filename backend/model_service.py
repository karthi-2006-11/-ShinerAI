"""
ShinerAI: Model Inference Service
Handles artifact loading (read-only), feature vector construction, and risk scoring.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd

from backend.config import (
    BASE_DIR,
    CONFIG_C_FEATURES,
    DECISION_THRESHOLD,
    DEFAULT_MODEL_PATH,
    METADATA_PATH,
    MODEL_ARTIFACT_PATH,
    OPERATIONAL_DO_THRESHOLD,
    PREDICTION_HORIZON_HOURS,
)


class ModelService:
    """Service for loading and executing trained ShinerAI model artifacts."""

    def __init__(self, artifact_path: Optional[str] = None):
        if artifact_path is None:
            artifact_path = MODEL_ARTIFACT_PATH

        self.artifact_path = Path(artifact_path).resolve()
        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.artifact_path}'. "
                f"Ensure Phase 3 artifacts exist in 'models/' directory."
            )

        # Load model artifact strictly read-only
        self.model = joblib.load(self.artifact_path)
        self.artifact_name = self.artifact_path.name
        self.model_family = self._determine_family()

        # Load metadata if present
        self.metadata = self._load_metadata()

    def _determine_family(self) -> str:
        """Determines model family from class type or artifact filename."""
        cls_name = self.model.__class__.__name__
        if "XGB" in cls_name or "xgb" in self.artifact_name.lower():
            return "XGBoost"
        elif "RandomForest" in cls_name or "rf" in self.artifact_name.lower():
            return "Random Forest"
        elif "Logistic" in cls_name or "lr" in self.artifact_name.lower():
            return "Logistic Regression"
        return cls_name

    def _load_metadata(self) -> Dict[str, Any]:
        """Loads model metadata from JSON file."""
        if METADATA_PATH.exists():
            try:
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def get_model_metrics(self) -> Dict[str, Any]:
        """Extracts documented test set metrics for this model from metadata."""
        models_meta = self.metadata.get("models", {})
        stem = self.artifact_path.stem.lower()

        # Check exact key or match stem
        if stem in models_meta:
            return models_meta[stem].get("metrics", {})

        # Fallback partial matching
        if "xgb" in stem and "config_c" in stem and "xgboost_config_c" in models_meta:
            return models_meta["xgboost_config_c"].get("metrics", {})
        if "rf" in stem and "config_c" in stem and "random_forest_config_c" in models_meta:
            return models_meta["random_forest_config_c"].get("metrics", {})

        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Returns structured metadata about the loaded model."""
        metrics = self.get_model_metrics()
        return {
            "api_name": "ShinerAI Early Warning API",
            "model_artifact": self.artifact_name,
            "model_family": self.model_family,
            "feature_configuration": "Config C (DO History Only)",
            "feature_count": len(CONFIG_C_FEATURES),
            "expected_features": CONFIG_C_FEATURES,
            "decision_threshold": DECISION_THRESHOLD,
            "operational_do_threshold": OPERATIONAL_DO_THRESHOLD,
            "prediction_horizon_hours": PREDICTION_HORIZON_HOURS,
            "held_out_test_metrics": metrics,
            "model_selection_guidance": (
                "XGBoost Config C achieves higher PR-AUC (0.7574) and Recall (79.75%), "
                "making it ideal when catching the maximum number of low-DO events is prioritized. "
                "Random Forest Config C achieves higher Specificity (93.11%) with fewer false alarms (504 vs 698), "
                "making it ideal when false alarm suppression is paramount. Model choice depends on farm operational trade-offs."
            ),
            "scientific_scope": (
                "Predictions quantify statistical risk associations of water DO falling below 3.0 mg/L "
                "within the 2-hour horizon. Predictions do NOT prove biological causality or directly predict fish mortality."
            ),
        }

    def predict(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes binary prediction for a single validated payload.

        Parameters
        ----------
        validated_data : dict
            Output from validation.validate_prediction_payload.

        Returns
        -------
        dict
            Prediction response dictionary.
        """
        pond_id = validated_data["pond_id"]
        prediction_timestamp = validated_data["prediction_timestamp"]
        features_dict = validated_data["features"]

        # Build single-row DataFrame with strict column order
        df = pd.DataFrame([features_dict], columns=CONFIG_C_FEATURES)

        # Predict probability for class 1 (AT_RISK) using numpy array
        probas = self.model.predict_proba(df.to_numpy())
        risk_probability = float(probas[0, 1])

        # Apply decision threshold
        is_at_risk = risk_probability >= DECISION_THRESHOLD
        predicted_label = "AT_RISK" if is_at_risk else "SAFE"
        binary_prediction = 1 if is_at_risk else 0

        # Create operator alert message
        current_do = features_dict["current_do"]
        if is_at_risk:
            alert_message = (
                f"WARNING: Pond '{pond_id}' is at risk of falling below {OPERATIONAL_DO_THRESHOLD:.1f} mg/L DO "
                f"within the next {PREDICTION_HORIZON_HOURS:.0f} hours (estimated risk probability: {risk_probability:.1%}, "
                f"current DO: {current_do:.2f} mg/L). Immediate inspection or aeration is recommended."
            )
        else:
            alert_message = (
                f"NORMAL: Pond '{pond_id}' is expected to remain safe (>= {OPERATIONAL_DO_THRESHOLD:.1f} mg/L DO) "
                f"over the next {PREDICTION_HORIZON_HOURS:.0f} hours (estimated risk probability: {risk_probability:.1%}, "
                f"current DO: {current_do:.2f} mg/L)."
            )

        return {
            "pond_id": pond_id,
            "prediction_timestamp": prediction_timestamp,
            "current_do": current_do,
            "risk_probability": round(risk_probability, 4),
            "predicted_label": predicted_label,
            "binary_prediction": binary_prediction,
            "decision_threshold": DECISION_THRESHOLD,
            "model_used": self.model_family,
            "model_artifact": self.artifact_name,
            "warning_issued": is_at_risk,
            "alert_message": alert_message,
        }
