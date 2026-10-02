# ShinerAI: Beginner's Guide to Phase 4
## Model Explainability and REST API Serving

**Project:** ShinerAI  
**Research Title:** AI-Based Early Warning System for Low Dissolved Oxygen in Fish Farms  
**Target Audience:** AI/ML Beginners and Developers  

---

## 1. What Did We Build in Phase 4?

In Phase 3, we trained Machine Learning models (like XGBoost and Random Forest) to predict whether dissolved oxygen (DO) in a fish pond will fall below $3.0\text{ mg/L}$ in the next 2 hours.

In **Phase 4**, we solved two important practical challenges:
1. **Explainability ("Why did the model say that?"):** Machine learning models can feel like "black boxes." We added **SHAP (SHapley Additive exPlanations)** so the model explains exactly *which features* pushed the risk up or down for any given prediction.
2. **Web API Serving ("How do other programs use the model?"):** We built a lightweight, modular **Flask REST API** that accepts pond sensor readings over HTTP, checks that the data is valid, runs the model, and returns risk predictions and explanations in JSON format.

---

## 2. Understanding Model Explainability (SHAP)

### What is SHAP?
SHAP is a game-theoretic approach to explain the output of any machine learning model. Think of each feature (like `current_do` or `do_t_minus_15`) as a player in a game:
- Some players work to push the prediction **toward AT_RISK** (positive SHAP value, colored red).
- Other players work to push the prediction **toward SAFE** (negative SHAP value, colored blue).
- The baseline (or expected value) is the average risk score. The SHAP values add up to explain the gap between the baseline and the final prediction!

### Two Types of Explainability:
1. **Global Feature Importance:**
   - *Question answered:* "Across the entire dataset, which features matter the most to the model overall?"
   - *Result:* `current_do` is #1, followed by diurnal time markers (`minute_of_day`, `hour_of_day`), followed by recent historical lags (`do_t_minus_30`, `do_t_minus_15`).
   - *Plots:* Saved in `results/figures/explainability/global_feature_importance_xgb_config_c.png`.
2. **Local Prediction Explanation:**
   - *Question answered:* "For this specific pond at 3:30 AM with DO = 3.84 mg/L, why did the model predict 93.4% risk?"
   - *Result:* We see that `current_do = 3.84` pushed risk up by $+2.08$, `minute_of_day = 210` pushed risk up by $+0.38$, and the rapid drop from $T-15\text{m}$ ($4.05\text{ mg/L}$) pushed risk up by $+0.15$.
   - *Plots:* Saved in `results/figures/explainability/local_explanation_at_risk_xgb_config_c.png`.

---

## 3. How to Run the REST API

### Step 1: Activate Virtual Environment
Open your terminal in the `d:\FISH` directory:
```powershell
.venv\Scripts\Activate.ps1
```

### Step 2: Start the Flask Server
Run the Flask backend directly:
```powershell
python -m backend.app
```
You will see output like:
```
 * Serving Flask app 'backend.app'
 * Debug mode: off
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
 * Running on http://<your-ip>:5000
```

### Switching Models via Environment Variable:
By default, the server loads `models/xgboost_config_c.joblib`. If you want to run with Random Forest instead:
```powershell
$env:MODEL_ARTIFACT_PATH="models/random_forest_config_c.joblib"
python -m backend.app
```

> [!NOTE]
> **Model Artifact Provenance:**  
> Config C model artifacts were reproduced using the frozen Phase 3 training procedure solely to create dedicated explainability/API artifacts; no model architecture, dataset, split, feature set, or training procedure was changed.

---

## 4. How to Send Requests to the API

You can test the API using PowerShell `curl.exe` or Python.

### Example 1: Check Server Health (`GET /health`)
```powershell
curl.exe http://localhost:5000/health
```
**Response:**
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

### Example 2: Make a Prediction (`POST /predict`)
Send a pond observation where oxygen is dropping at night (using canonical `do_t_minus_X` keys):
```powershell
curl.exe -X POST http://localhost:5000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "pond_id": "pond_nocturnal_01",
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

**Response:**
```json
{
  "alert_message": "WARNING: Pond 'pond_nocturnal_01' is at risk of falling below 3.0 mg/L DO within the next 2 hours (estimated risk probability: 93.4%, current DO: 3.84 mg/L). Immediate inspection or aeration is recommended.",
  "binary_prediction": 1,
  "current_do": 3.84,
  "decision_threshold": 0.5,
  "model_artifact": "xgboost_config_c.joblib",
  "model_used": "XGBoost",
  "pond_id": "pond_nocturnal_01",
  "predicted_label": "AT_RISK",
  "prediction_timestamp": "2026-01-26T03:30:00",
  "risk_probability": 0.9341,
  "warning_issued": true
}
```

---

### Example 3: Get Prediction with SHAP Explanation (`POST /explain`)
Send the same data to `/explain` to see the reasons behind the prediction:
```powershell
curl.exe -X POST http://localhost:5000/explain `
  -H "Content-Type: application/json" `
  -d '{
    "pond_id": "pond_nocturnal_01",
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

**Response includes `top_risk_drivers`:**
```json
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
]
```

---

## 5. Common Questions & Pitfalls for Beginners

### Q1: Why do I get a 400 error when `current_do` is 2.5 mg/L?
**Answer:** The goal of ShinerAI is **early warning**. It is designed to alert fish farmers *before* dissolved oxygen drops below $3.0\text{ mg/L}$. If current DO is already $2.5\text{ mg/L}$, the pond is already in a critical state! The system rejects this request with HTTP `400 Bad Request` and `status: ALREADY_LOW_DO` because predicting future risk is unnecessary—emergency aeration should already be running.

### Q2: What is the canonical naming format for historical lags?
**Answer:** The canonical format is `do_t_minus_15` through `do_t_minus_120`. While the API accepts shorter aliases like `do_t-15m` internally for convenience, we recommend using the canonical `do_t_minus_X` format in all integrations to match the model feature schema.

### Q3: Do I need to calculate `hour_of_day` and `minute_of_day` myself?
**Answer:** No! Just pass an ISO timestamp like `"2026-01-26T03:30:00"`. The API automatically parses the timestamp and extracts the exact hour and minute for the model.

### Q4: Does SHAP prove that dropping DO caused the fish to die?
**Answer:** **No.** This is a critical scientific distinction:
1. SHAP measures **statistical association**, not biological causation.
2. The model predicts **water oxygen level** ($\text{DO} < 3.0\text{ mg/L}$), not fish mortality.
3. Keep your scientific reports accurate by stating that the model detects patterns associated with water deoxygenation.
