"""
Unit and integration tests for the ShinerAI Flask REST API (Phase 4).
Tests endpoints: GET /, GET /health, GET /model-info, POST /predict, POST /explain,
and verifies strict boundary checks, error handling, and model configurability.
"""

import pytest
from backend.app import create_app
from backend.config import CONFIG_C_FEATURES, OPERATIONAL_DO_THRESHOLD


@pytest.fixture
def client_xgb():
    """Flask test client using default XGBoost Config C model."""
    app = create_app("models/xgboost_config_c.joblib")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def client_rf():
    """Flask test client using Random Forest Config C model."""
    app = create_app("models/random_forest_config_c.joblib")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def valid_safe_payload():
    """Valid observation where DO is ascending during morning hours (SAFE)."""
    return {
        "pond_id": "ara2_0677080b",
        "prediction_timestamp": "2026-01-26T10:15:00",
        "current_do": 5.40,
        "do_t_minus_15": 5.15,
        "do_t_minus_30": 4.87,
        "do_t_minus_45": 4.36,
        "do_t_minus_60": 4.21,
        "do_t_minus_75": 3.89,
        "do_t_minus_90": 3.72,
        "do_t_minus_105": 3.23,
        "do_t_minus_120": 2.92,
    }


@pytest.fixture
def valid_at_risk_payload():
    """Valid observation where DO is declining at night toward 3.0 (AT_RISK)."""
    return {
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
        "do_t_minus_120": 5.02,
    }


# ==============================================================================
# Endpoint: GET /
# ==============================================================================

def test_index_endpoint(client_xgb):
    response = client_xgb.get("/")
    assert response.status_code == 200
    data = response.get_json()
    assert data["project"] == "ShinerAI"
    assert data["status"] == "online"
    assert "endpoints" in data
    assert any("/health" in k for k in data["endpoints"])
    assert any("/predict" in k for k in data["endpoints"])


# ==============================================================================
# Endpoint: GET /health
# ==============================================================================

def test_health_endpoint_xgb(client_xgb):
    response = client_xgb.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["project"] == "ShinerAI"
    assert data["model_loaded"] is True
    assert data["model_family"] == "XGBoost"
    assert "timestamp" in data


def test_health_endpoint_rf(client_rf):
    response = client_rf.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_family"] == "Random Forest"


# ==============================================================================
# Endpoint: GET /model-info
# ==============================================================================

def test_model_info_endpoint(client_xgb):
    response = client_xgb.get("/model-info")
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_artifact"] == "xgboost_config_c.joblib"
    assert data["model_family"] == "XGBoost"
    assert data["feature_count"] == 11
    assert data["expected_features"] == CONFIG_C_FEATURES
    assert data["decision_threshold"] == 0.50
    assert data["operational_do_threshold"] == OPERATIONAL_DO_THRESHOLD
    assert "held_out_test_metrics" in data
    assert data["held_out_test_metrics"]["pr_auc"] == 0.7574
    assert "scientific_scope" in data


# ==============================================================================
# Endpoint: POST /predict (Valid Cases)
# ==============================================================================

def test_predict_safe_observation(client_xgb, valid_safe_payload):
    response = client_xgb.post("/predict", json=valid_safe_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["pond_id"] == "ara2_0677080b"
    assert data["predicted_label"] == "SAFE"
    assert data["binary_prediction"] == 0
    assert data["warning_issued"] is False
    assert 0.0 <= data["risk_probability"] < 0.50
    assert "NORMAL" in data["alert_message"]


def test_predict_at_risk_observation(client_xgb, valid_at_risk_payload):
    response = client_xgb.post("/predict", json=valid_at_risk_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["pond_id"] == "ara2_0677080b"
    assert data["predicted_label"] == "AT_RISK"
    assert data["binary_prediction"] == 1
    assert data["warning_issued"] is True
    assert 0.50 <= data["risk_probability"] <= 1.0
    assert "WARNING" in data["alert_message"]


def test_predict_with_lag_aliases(client_xgb):
    """Test that client can pass 'do_t-15m' style aliases instead of 'do_t_minus_15'."""
    alias_payload = {
        "pond_id": "pond_alias_test",
        "prediction_timestamp": "2026-01-26 03:30:00",
        "current_do": 3.84,
        "do_t-15m": 4.05,
        "do_t-30m": 4.49,
        "do_t-45m": 4.67,
        "do_t-60m": 4.69,
        "do_t-75m": 4.73,
        "do_t-90m": 4.85,
        "do_t-105m": 5.00,
        "do_t-120m": 5.02,
    }
    response = client_xgb.post("/predict", json=alias_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["predicted_label"] == "AT_RISK"
    assert data["warning_issued"] is True


def test_predict_rf_model(client_rf, valid_at_risk_payload):
    """Verify inference behaves consistently with Random Forest Config C."""
    response = client_rf.post("/predict", json=valid_at_risk_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_used"] == "Random Forest"
    assert data["predicted_label"] == "AT_RISK"
    assert data["risk_probability"] > 0.50


# ==============================================================================
# Endpoint: POST /predict (Validation & Error Cases)
# ==============================================================================

def test_predict_boundary_condition_current_do_below_3(client_xgb, valid_at_risk_payload):
    """Operational rule: current_do < 3.0 must return 400 Bad Request."""
    payload = valid_at_risk_payload.copy()
    payload["current_do"] = 2.85
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "below the 3.0 mg/L threshold" in data["message"]
    assert data["details"]["status"] == "ALREADY_LOW_DO"


def test_predict_missing_current_do(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    del payload["current_do"]
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "current_do is required" in data["message"]


def test_predict_negative_current_do(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    payload["current_do"] = -1.5
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    data = response.get_json()
    assert data["error"] == "ValidationError"
    assert "cannot be negative" in data["message"]


def test_predict_missing_pond_id(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    payload["pond_id"] = "   "
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    assert "pond_id is required" in response.get_json()["message"]


def test_predict_invalid_timestamp(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    payload["prediction_timestamp"] = "not-a-valid-time"
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Cannot parse prediction_timestamp" in response.get_json()["message"]


def test_predict_missing_lag_feature(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    del payload["do_t_minus_120"]
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Missing required historical lag features" in response.get_json()["message"]


def test_predict_non_numeric_lag(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    payload["do_t_minus_30"] = "corrupted_sensor_string"
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid lag features" in response.get_json()["message"]


def test_predict_negative_lag(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    payload["do_t_minus_60"] = -2.0
    response = client_xgb.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Invalid lag features" in response.get_json()["message"]


def test_predict_empty_or_non_json_body(client_xgb):
    response = client_xgb.post("/predict", data="plain text not json", content_type="text/plain")
    assert response.status_code == 400


# ==============================================================================
# Endpoint: POST /explain
# ==============================================================================

def test_explain_endpoint_safe(client_xgb, valid_safe_payload):
    response = client_xgb.post("/explain", json=valid_safe_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["predicted_label"] == "SAFE"
    assert "feature_contributions" in data
    assert len(data["feature_contributions"]) == 11
    assert "base_value" in data
    assert "top_safe_drivers" in data
    assert len(data["top_safe_drivers"]) > 0
    assert "scientific_note" in data
    assert "not prove biological causality" in data["scientific_note"]


def test_explain_endpoint_at_risk(client_xgb, valid_at_risk_payload):
    response = client_xgb.post("/explain", json=valid_at_risk_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["predicted_label"] == "AT_RISK"
    assert "feature_contributions" in data
    assert "top_risk_drivers" in data
    assert len(data["top_risk_drivers"]) > 0
    # Current DO should be a primary driver
    top_driver_features = [d["feature"] for d in data["top_risk_drivers"]]
    assert "current_do" in top_driver_features


def test_explain_endpoint_rf(client_rf, valid_at_risk_payload):
    response = client_rf.post("/explain", json=valid_at_risk_payload)
    assert response.status_code == 200
    data = response.get_json()
    assert data["model_used"] == "Random Forest"
    assert len(data["feature_contributions"]) == 11
    assert "base_value" in data


def test_explain_boundary_condition(client_xgb, valid_at_risk_payload):
    payload = valid_at_risk_payload.copy()
    payload["current_do"] = 2.50
    response = client_xgb.post("/explain", json=payload)
    assert response.status_code == 400
    assert "below the 3.0 mg/L threshold" in response.get_json()["message"]


# ==============================================================================
# HTTP Method and Not Found Handling
# ==============================================================================

def test_not_found_endpoint(client_xgb):
    response = client_xgb.get("/non_existent_route")
    assert response.status_code == 404
    data = response.get_json()
    assert data["error"] == "NotFound"


def test_method_not_allowed(client_xgb):
    response = client_xgb.post("/health")
    assert response.status_code == 405
    data = response.get_json()
    assert data["error"] == "MethodNotAllowed"
