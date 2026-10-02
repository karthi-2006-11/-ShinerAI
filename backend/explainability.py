"""
ShinerAI: Model Explainability Service
Computes local SHAP TreeExplainer attributions for single-instance REST API requests.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import shap

from backend.config import CONFIG_C_FEATURES


class ExplainabilityService:
    """Provides on-demand local SHAP explanations for tree-based models."""

    def __init__(self, model: Any):
        self.model = model
        self.explainer: Optional[shap.TreeExplainer] = None

    def _get_explainer(self) -> shap.TreeExplainer:
        """Lazily instantiates and caches the TreeExplainer."""
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        return self.explainer

    def explain(self, validated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Computes local SHAP attributions for a single observation.

        Parameters
        ----------
        validated_data : dict
            Output from validation.validate_prediction_payload.

        Returns
        -------
        dict
            Structured SHAP explanation including feature contributions,
            top risk drivers, top safe drivers, base value, and scientific notes.
        """
        features_dict = validated_data["features"]
        df = pd.DataFrame([features_dict], columns=CONFIG_C_FEATURES)

        explainer = self._get_explainer()
        explanation = explainer(df)

        raw_values = explanation.values[0]
        raw_base = explanation.base_values[0]

        # Handle multi-class (e.g., RandomForest binary output shape [11, 2])
        if hasattr(raw_values, "ndim") and raw_values.ndim == 2:
            shap_vals = raw_values[:, 1]
            base_value = float(raw_base[1]) if hasattr(raw_base, "__len__") else float(raw_base)
        else:
            shap_vals = raw_values
            base_value = float(raw_base) if not hasattr(raw_base, "__len__") else float(raw_base[0])

        feature_contributions: List[Dict[str, Any]] = []
        for i, col in enumerate(CONFIG_C_FEATURES):
            val = float(df.iloc[0][col])
            sv = float(shap_vals[i])
            direction = "toward_AT_RISK" if sv > 0 else "toward_SAFE"
            feature_contributions.append({
                "feature": col,
                "value": round(val, 4),
                "shap_value": round(sv, 4),
                "direction": direction,
            })

        # Sort all contributions by magnitude (absolute SHAP value)
        sorted_contributions = sorted(
            feature_contributions,
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        # Extract top risk drivers (positive SHAP pushing toward AT_RISK)
        top_risk_drivers = [
            f for f in sorted_contributions if f["shap_value"] > 0
        ]

        # Extract top safe drivers (negative SHAP pushing toward SAFE)
        top_safe_drivers = [
            f for f in sorted_contributions if f["shap_value"] < 0
        ]

        return {
            "base_value": round(base_value, 4),
            "feature_contributions": sorted_contributions,
            "top_risk_drivers": top_risk_drivers[:3],
            "top_safe_drivers": top_safe_drivers[:3],
            "scientific_note": (
                "SHAP values quantify statistical feature attributions within the trained model and dataset; "
                "they do not prove biological causality. Historical inputs are discrete lag observations; "
                "no explicit rate-of-change derivative was engineered. "
                "ShinerAI forecasts water oxygen depletion (< 3.0 mg/L in 2 hours), not fish mortality."
            ),
        }
