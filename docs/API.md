# ShinerAI: REST API Specification (Phase 4)

**Project:** ShinerAI  
**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Base URL:** `http://localhost:5000` (Default local development)  
**Protocol:** HTTP/1.1  
**Data Format:** JSON (`Content-Type: application/json`)  
**Active Default Model:** `models/xgboost_config_c.joblib` (Configurable via `MODEL_ARTIFACT_PATH`)

---

## 1. Overview & Operational Problem Definition

The ShinerAI REST API provides early warning inference and explainability for continuous fish-pond water quality monitoring:
> **Operational Problem:** Given an individual fish-farm pond with current dissolved oxygen ($\text{DO} \ge 3.0\text{ mg/L}$) and recent 2-hour historical trajectory ($t-15\text{m} \dots t-120\text{m}$), will dissolved oxygen fall below the critical hypoxic threshold ($3.0\text{ mg/L}$) within the next 2 hours?

### Input Constraint:
- The system operates strictly when current DO is at or above $3.0\text{ mg/L}$.
- If current DO is already $< 3.0\text{ mg/L}$, the API rejects the request with HTTP `400 Bad Request` and `status: ALREADY_LOW_DO`, because the pond is already experiencing low DO and predictive early warning is inapplicable.

---

## 2. API Endpoints

### 2.1 Index Directory: `GET /`
Provides API discovery, status, active model artifact, and available endpoints.

#### Response (`200 OK`):
```json
{
  "project": "ShinerAI",
  "title": "AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms",
  "status": "online",
  "phase": "Phase 4 - Model Serving & Explainability",
  "active_model": "xgboost_config_c.joblib",
  "endpoints": {
    "GET /health": "Server health, model load status, and runtime environment.",
    "GET /model-info": "Model architecture, feature schema, decision threshold, and held-out test metrics.",
    "POST /predict": "Predict whether pond DO will drop below 3.0 mg/L within 2 hours.",
    "POST /explain": "Generate prediction along with local SHAP feature attributions."
  }
}
```

---

### 2.2 Health Check: `GET /health`
Returns server operational status, active model artifact, model family, decision threshold, and server UTC timestamp.

#### Response (`200 OK`):
```json
{
  "status": "healthy",
  "project": "ShinerAI",
  "model_loaded": true,
  "model_artifact": "xgboost_config_c.joblib",
  "model_family": "XGBoost",
  "decision_threshold": 0.5,
  "timestamp": "2026-10-02T07:23:05.124347+00:00"
}
```

---

### 2.3 Model Information: `GET /model-info`
Provides detailed specifications of the loaded model, including expected features, operational thresholds, held-out test metrics, trade-off guidance, and scientific framing.

#### Response (`200 OK`):
```json
{
  "api_name": "ShinerAI Early Warning API",
  "decision_threshold": 0.5,
  "expected_features": [
    "current_do",
    "hour_of_day",
    "minute_of_day",
    "do_t_minus_15",
    "do_t_minus_30",
    "do_t_minus_45",
    "do_t_minus_60",
    "do_t_minus_75",
    "do_t_minus_90",
    "do_t_minus_105",
    "do_t_minus_120"
  ],
  "feature_configuration": "Config C (DO History Only)",
  "feature_count": 11,
  "held_out_test_metrics": {
    "accuracy": 0.8924,
    "f1": 0.6285,
    "fn": 191,
    "fp": 698,
    "pr_auc": 0.7574,
    "precision": 0.5186,
    "recall": 0.7975,
    "roc_auc": 0.9162,
    "specificity": 0.9046,
    "test_examples": 8261,
    "test_neg": 7318,
    "test_pos": 943,
    "threshold": 0.5,
    "threshold": 0.5,
    "tn": 6620,
    "tp": 752
  },
  "model_artifact": "xgboost_config_c.joblib",
  "model_family": "XGBoost",
  "model_selection_guidance": "XGBoost Config C achieves the highest PR-AUC among tested models (0.7574) and highest recall among Config C tree models (79.75%), making it preferable when prioritizing detection of low-DO events within the DO-only feature space. Random Forest Config C achieves higher Specificity (93.11%), producing 194 fewer false positives than XGBoost Config C at the default threshold; this could reduce unnecessary interventions in a deployment where alerts trigger aeration. Model choice depends on farm operational trade-offs.",
  "operational_do_threshold": 3.0,
  "prediction_horizon_hours": 2.0,
  "scientific_scope": "Predictions quantify statistical risk associations of water DO falling below 3.0 mg/L within the 2-hour horizon. Predictions do NOT prove biological causality or directly predict fish mortality."
}
```

---

### 2.4 Prediction: `POST /predict`
Executes risk scoring for a single pond observation.

#### Request Headers:
`Content-Type: application/json`

#### Request Payload:
| Field | Type | Required | Description |
|:---|:---|:---:|:---|
| `pond_id` | string | Yes | Unique pond identifier (e.g., `"ara2_0677080b"`). |
| `prediction_timestamp` | string | Yes | ISO 8601 timestamp (e.g., `"2026-01-26T03:30:00"`). |
| `current_do` | float | Yes | Current DO in mg/L. **Must be $\ge 3.0$**. |
| `do_t_minus_15` | float | Yes | DO reading 15 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_30` | float | Yes | DO reading 30 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_45` | float | Yes | DO reading 45 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_60` | float | Yes | DO reading 60 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_75` | float | Yes | DO reading 75 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_90` | float | Yes | DO reading 90 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_105` | float | Yes | DO reading 105 minutes prior (mg/L). **Canonical format**. |
| `do_t_minus_120` | float | Yes | DO reading 120 minutes prior (mg/L). **Canonical format**. |

> **Canonical Naming Convention:**  
> The canonical public standard is `do_t_minus_15` through `do_t_minus_120` (matching the model feature schema). While alternative short aliases like `do_t-15m` are accepted internally by the server for client compatibility, all client integrations and documentation should use the canonical `do_t_minus_X` format.

*Note: Temporal features (`hour_of_day` and `minute_of_day`) are automatically extracted from `prediction_timestamp`.*

#### Example Request:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pond_id": "ara2_0677080b",
    "prediction_timestamp": "2026-01-26T03:30:00",
    "current_do": 3.84,
    "do_t_minus_15": 4.05,
    "do_t_minus_30": 4.49,
    "do_t_minus_45": 4.67,
    "do_t_minus_60": 4.69,
    "do_t_minus_75": 4.73,
    "do_t_minus_90": 4.85,
    "do_t_minus_105": 5.00,
    "do_t_minus_120": 5.02
  }'
```

#### Response (`200 OK`):
```json
{
  "alert_message": "WARNING: Pond 'ara2_0677080b' is at risk of falling below 3.0 mg/L DO within the next 2 hours (estimated risk probability: 93.4%, current DO: 3.84 mg/L). Immediate inspection or aeration is recommended.",
  "binary_prediction": 1,
  "current_do": 3.84,
  "decision_threshold": 0.5,
  "model_artifact": "xgboost_config_c.joblib",
  "model_used": "XGBoost",
  "pond_id": "ara2_0677080b",
  "predicted_label": "AT_RISK",
  "prediction_timestamp": "2026-01-26T03:30:00",
  "risk_probability": 0.9341,
  "warning_issued": true
}
```

---

### 2.5 Explainability: `POST /explain`
Executes risk scoring and generates local SHAP feature attributions decomposing the prediction.

#### Request:
Same input payload format as `POST /predict`.

#### Response (`200 OK`):
```json
{
  "alert_message": "WARNING: Pond 'ara2_0677080b' is at risk of falling below 3.0 mg/L DO within the next 2 hours (estimated risk probability: 93.4%, current DO: 3.84 mg/L). Immediate inspection or aeration is recommended.",
  "base_value": -0.0039,
  "binary_prediction": 1,
  "current_do": 3.84,
  "decision_threshold": 0.5,
  "feature_contributions": [
    {
      "direction": "toward_AT_RISK",
      "feature": "current_do",
      "shap_value": 2.0758,
      "value": 3.84
    },
    {
      "direction": "toward_AT_RISK",
      "feature": "minute_of_day",
      "shap_value": 0.3809,
      "value": 210.0
    },
    {
      "direction": "toward_AT_RISK",
      "feature": "do_t_minus_15",
      "shap_value": 0.1468,
      "value": 4.05
    },
    {
      "direction": "toward_SAFE",
      "feature": "do_t_minus_120",
      "shap_value": -0.109,
      "value": 5.02
    }
  ],
  "model_artifact": "xgboost_config_c.joblib",
  "model_used": "XGBoost",
  "pond_id": "ara2_0677080b",
  "predicted_label": "AT_RISK",
  "prediction_timestamp": "2026-01-26T03:30:00",
  "risk_probability": 0.9341,
  "scientific_note": "SHAP values quantify statistical feature attributions within the trained model and dataset; they do not prove biological causality. Historical inputs are discrete lag observations; no explicit rate-of-change derivative was engineered. ShinerAI forecasts water oxygen depletion (< 3.0 mg/L in 2 hours), not fish mortality.",
  "top_risk_drivers": [
    {
      "direction": "toward_AT_RISK",
      "feature": "current_do",
      "shap_value": 2.0758,
      "value": 3.84
    },
    {
      "direction": "toward_AT_RISK",
      "feature": "minute_of_day",
      "shap_value": 0.3809,
      "value": 210.0
    },
    {
      "direction": "toward_AT_RISK",
      "feature": "do_t_minus_15",
      "shap_value": 0.1468,
      "value": 4.05
    }
  ],
  "top_safe_drivers": [
    {
      "direction": "toward_SAFE",
      "feature": "do_t_minus_120",
      "shap_value": -0.109,
      "value": 5.02
    }
  ],
  "warning_issued": true
}
```

---

## 3. Error Responses & Status Codes

All errors return JSON with structured, informative messages:

| HTTP Status | Error Type | Description |
|:---:|:---|:---|
| `400 Bad Request` | `ValidationError` | Missing required fields, unparseable timestamp, negative values, or operational boundary violation (`current_do < 3.0`). |
| `400 Bad Request` | `BadRequest` | Malformed or non-JSON body. |
| `404 Not Found` | `NotFound` | Non-existent route. |
| `405 Method Not Allowed` | `MethodNotAllowed` | Disallowed HTTP verb. |
| `500 Internal Server Error` | `InternalServerError` | Server-side execution exception. |

### Operational Boundary Violation Example (`400 Bad Request`):
```json
{
  "details": {
    "current_do": 2.7,
    "minimum_operational_threshold": 3.0,
    "status": "ALREADY_LOW_DO"
  },
  "error": "ValidationError",
  "field": "current_do",
  "message": "Operational boundary condition violated: current_do is already 2.70 mg/L (below the 3.0 mg/L threshold). ShinerAI operates only when current DO is at or above 3.0 mg/L to predict low-DO risk within the next 2 hours."
}
```
