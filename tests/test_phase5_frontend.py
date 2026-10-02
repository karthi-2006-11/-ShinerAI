"""
Phase 5 Tests: Dashboard Serving, UI Components, and End-to-End Integration.
Verifies:
1. Static dashboard delivery (HTML, CSS, JS) via Flask.
2. HTML structure, required form inputs, and accessibility markers.
3. Demo SAFE and AT_RISK scenario payloads and their end-to-end API predictions.
4. Client-side and server-side boundary condition validation (current_do < 3.0).
5. 2-Hour DO trajectory chart reference line (3.0 mg/L) and data mapping.
6. SHAP explanation rendering consistency and direction attribution.
7. Clean error handling without stack trace leakage.
8. Compliance with scientific language guardrails.
"""

from pathlib import Path
import re
import pytest

from backend.app import create_app
from backend.config import OPERATIONAL_DO_THRESHOLD


@pytest.fixture
def client():
    """Flask test client using default XGBoost Config C model."""
    app = create_app("models/xgboost_config_c.joblib")
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def demo_safe_payload():
    """Exact evaluated holdout observation for Daytime Recovery (SAFE)."""
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
def demo_at_risk_payload():
    """Exact evaluated holdout observation for Nocturnal Depletion (AT_RISK)."""
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
# 1. Dashboard Serving & Static Asset Tests
# ==============================================================================

def test_dashboard_served_via_index_with_html_accept(client):
    """Browser request to GET / with text/html Accept header receives dashboard HTML."""
    response = client.get("/", headers={"Accept": "text/html,application/xhtml+xml"})
    assert response.status_code == 200
    assert "text/html" in response.content_type
    html = response.get_data(as_text=True)
    assert "<title>ShinerAI" in html
    assert "AI-Based Early Warning System" in html


def test_dashboard_served_via_dashboard_route(client):
    """GET /dashboard serves the dashboard HTML."""
    response = client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.content_type
    html = response.get_data(as_text=True)
    assert "pondInputForm" in html


def test_static_stylesheet_served(client):
    """GET /style.css serves valid CSS with design tokens."""
    response = client.get("/style.css")
    assert response.status_code == 200
    assert "text/css" in response.content_type
    css = response.get_data(as_text=True)
    assert "--brand-primary" in css
    assert "--safe-primary" in css
    assert "--risk-primary" in css


def test_static_javascript_served(client):
    """GET /app.js serves valid JS controller."""
    response = client.get("/app.js")
    assert response.status_code == 200
    assert "javascript" in response.content_type
    js = response.get_data(as_text=True)
    assert "checkApiHealth" in js
    assert "renderDoChart" in js
    assert "DEMO_SCENARIOS" in js


def test_static_aquarium_javascript_served(client):
    """GET /aquarium.js serves valid aquarium animation engine."""
    response = client.get("/aquarium.js")
    assert response.status_code == 200
    assert "javascript" in response.content_type
    js = response.get_data(as_text=True)
    assert "Goldfish" in js
    assert "FishGroup" in js
    assert "aquarium-canvas" in js


# ==============================================================================
# 2. HTML Structure & Accessibility Tests
# ==============================================================================

def test_html_contains_all_required_form_fields():
    """Verify frontend/index.html includes all required telemetry and metadata inputs."""
    html_path = Path("frontend/index.html")
    assert html_path.exists(), "frontend/index.html must exist"
    html = html_path.read_text(encoding="utf-8")

    # Metadata fields
    assert 'id="pondId"' in html
    assert 'id="predictionTimestamp"' in html

    # Current DO
    assert 'id="currentDo"' in html

    # 8 historical lags
    expected_lags = [
        "do_t_minus_15",
        "do_t_minus_30",
        "do_t_minus_45",
        "do_t_minus_60",
        "do_t_minus_75",
        "do_t_minus_90",
        "do_t_minus_105",
        "do_t_minus_120",
    ]
    for lag in expected_lags:
        assert f'id="{lag}"' in html, f"Missing lag input field {lag}"

    # Action buttons
    assert 'id="btnAnalyze"' in html
    assert 'id="btnDemoSafe"' in html
    assert 'id="btnDemoAtRisk"' in html
    assert 'id="btnClearInputs"' in html

    # Visual panels
    assert 'id="apiStatusBadge"' in html
    assert 'id="doChart"' in html
    assert 'id="riskBanner"' in html
    assert 'id="riskProbabilityValue"' in html
    assert 'id="shapTableBody"' in html
    assert 'id="globalImportanceChart"' in html

    # Demo scenario labels
    assert 'SAFE Case Study — Rising DO During Daytime' in html
    assert 'AT_RISK Case Study — Declining DO During Nighttime' in html

    # Model specifications table
    assert '0.9162' in html
    assert '0.7574' in html
    assert '0.9458' not in html
    assert '0.7718' not in html


def test_html_accessibility_elements():
    """Verify labels, ARIA landmarks, and accessibility semantics."""
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert 'role="banner"' in html
    assert 'role="main"' in html
    assert 'role="contentinfo"' in html
    assert 'aria-label' in html or 'aria-labelledby' in html
    # Check that labels associate with inputs
    assert '<label for="pondId">' in html
    assert '<label for="currentDo">' in html
    # Aquarium visual background must be marked aria-hidden for screen readers
    assert 'id="aquarium-background" aria-hidden="true"' in html
    assert 'id="aquarium-canvas"' in html


# ==============================================================================
# 3. Demo Scenarios & End-to-End Prediction Tests
# ==============================================================================

def test_demo_safe_scenario_execution(client, demo_safe_payload):
    """End-to-end verification of Daytime Recovery scenario produces SAFE result."""
    response = client.post("/predict", json=demo_safe_payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["predicted_label"] == "SAFE"
    assert data["binary_prediction"] == 0
    assert data["warning_issued"] is False
    assert data["risk_probability"] < 0.50
    assert 0.0 <= data["risk_probability"] <= 0.20  # Expected ~6.7% risk


def test_demo_at_risk_scenario_execution(client, demo_at_risk_payload):
    """End-to-end verification of Nocturnal Depletion scenario produces AT_RISK result."""
    response = client.post("/predict", json=demo_at_risk_payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["predicted_label"] == "AT_RISK"
    assert data["binary_prediction"] == 1
    assert data["warning_issued"] is True
    assert data["risk_probability"] >= 0.50
    assert data["risk_probability"] > 0.85  # Expected ~93.4% risk


# ==============================================================================
# 4. End-to-End Explainability Integration Tests
# ==============================================================================

def test_explain_endpoint_safe_scenario(client, demo_safe_payload):
    """Explain endpoint returns structured SHAP attributions for SAFE scenario."""
    response = client.post("/explain", json=demo_safe_payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["predicted_label"] == "SAFE"
    assert "feature_contributions" in data
    assert len(data["feature_contributions"]) == 11
    assert "top_safe_drivers" in data
    assert len(data["top_safe_drivers"]) > 0

    # Ensure each contribution has feature, value, shap_value, direction
    for item in data["feature_contributions"]:
        assert "feature" in item
        assert "value" in item
        assert "shap_value" in item
        assert item["direction"] in ["toward_SAFE", "toward_AT_RISK"]


def test_explain_endpoint_at_risk_scenario(client, demo_at_risk_payload):
    """Explain endpoint returns structured SHAP attributions for AT_RISK scenario."""
    response = client.post("/explain", json=demo_at_risk_payload)
    assert response.status_code == 200
    data = response.get_json()

    assert data["predicted_label"] == "AT_RISK"
    assert "top_risk_drivers" in data
    assert len(data["top_risk_drivers"]) > 0


# ==============================================================================
# 5. Boundary Condition & Error Handling Tests
# ==============================================================================

def test_operational_boundary_rejection_below_threshold(client, demo_safe_payload):
    """Current DO below 3.0 mg/L rejected with HTTP 400 ALREADY_LOW_DO."""
    invalid_payload = {**demo_safe_payload, "current_do": 2.85}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    data = response.get_json()

    assert data["error"] == "ValidationError"
    assert data["field"] == "current_do"
    assert data["details"]["status"] == "ALREADY_LOW_DO"
    assert "Operational boundary condition violated" in data["message"]


def test_clean_error_handling_no_traceback_exposed(client):
    """Malformed payload returns structured JSON error without Python traceback."""
    response = client.post("/predict", data="Not a JSON", content_type="application/json")
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "Traceback" not in response.get_data(as_text=True)


# ==============================================================================
# 6. Chart Reference Line & Visual Logic Tests
# ==============================================================================

def test_chart_reference_line_present_in_code():
    """Verify that 3.0 mg/L reference line is implemented in frontend code."""
    app_js = Path("frontend/app.js").read_text(encoding="utf-8")
    assert "OPERATIONAL_DO_THRESHOLD = 3.0" in app_js
    assert "3.0 mg/L (Provisional Threshold)" in app_js or "3.0 mg/L" in app_js
    assert "stroke-dasharray" in app_js

    index_html = Path("frontend/index.html").read_text(encoding="utf-8")
    assert "3.0 mg/L" in index_html
    assert "provisional project threshold" in index_html


def test_global_feature_importance_matches_phase4_artifact():
    """Verify frontend global feature importance matches XGBoost Config C artifact."""
    app_js = Path("frontend/app.js").read_text(encoding="utf-8")
    # current_do must be rank 1 with ~1.3896
    assert "current_do" in app_js
    assert "1.3896" in app_js
    # minute_of_day rank 2 with ~0.5388
    assert "minute_of_day" in app_js
    assert "0.5388" in app_js


# ==============================================================================
# 7. Scientific Language Guardrails Tests
# ==============================================================================

def test_scientific_language_guardrails_in_frontend():
    """Ensure frontend text adheres to strict scientific language guardrails."""
    html = Path("frontend/index.html").read_text(encoding="utf-8")
    js = Path("frontend/app.js").read_text(encoding="utf-8")

    forbidden_phrases = [
        "prevents fish death",
        "predicts fish mortality",
        "causes oxygen depletion",
        "proves biological mechanisms",
        "100% accurate",
        "guarantees hypoxia prevention",
        "guaranteed",
    ]

    for phrase in forbidden_phrases:
        assert phrase not in html.lower(), f"Forbidden phrase '{phrase}' found in index.html"
        assert phrase not in js.lower(), f"Forbidden phrase '{phrase}' found in app.js"

    # Confirm presence of required caveats
    assert "does not predict fish disease" in html.lower() or "not predict fish disease" in html.lower()
    assert "provisional" in html.lower()
    assert "model behavior, not biological causation" in html.lower() or "not biological causation" in html.lower()


def test_health_connection_check(client):
    """Verify /health endpoint returns healthy status and model load confirmation."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["project"] == "ShinerAI"


def test_timestamp_derives_hour_and_minute(client, demo_safe_payload):
    """Verify that timestamp correctly derives hour and minute of day in inference."""
    response = client.post("/explain", json=demo_safe_payload)
    assert response.status_code == 200
    data = response.get_json()
    contrib_features = {c["feature"]: c["value"] for c in data["feature_contributions"]}
    # '2026-01-26T10:15:00' -> hour 10, minute 10*60 + 15 = 615
    assert contrib_features["hour_of_day"] == 10.0
    assert contrib_features["minute_of_day"] == 615.0


def test_model_info_metrics_consistency(client):
    """Verify /model-info returns exact frozen Phase 3 metrics (ROC-AUC 0.9162, PR-AUC 0.7574)."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.get_json()
    metrics = data["held_out_test_metrics"]
    assert metrics["roc_auc"] == 0.9162
    assert metrics["pr_auc"] == 0.7574


