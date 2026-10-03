# ShinerAI: Feature Ablation Comparison (Value of Historical Telemetry)

This ablation analysis directly addresses mentor feedback by quantifying whether **recent dissolved-oxygen history adds predictive information** beyond current-time measurements alone.

### Feature Configurations Evaluated:
- **Config A (Current Only, 5 features):** `current_do`, `current_ph`, `current_temperature`, `hour_of_day`, `minute_of_day`.
- **Config B (Current + History, 29 features):** Current values + 8 lags each of DO, pH, and Temperature ($t-15\text{m}$ to $t-120\text{m}$) + diurnal time features.
- **Config C (DO History Only, 11 features):** `current_do` + 8 historical DO lags ($t-15\text{m}$ to $t-120\text{m}$) + diurnal time features.

### Ablation Results (PR-AUC Metric):

| Model | Config A PR-AUC | Config B PR-AUC | Config C PR-AUC | Config B - Config A | Config C - Config A | Config C - Config B |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.6019 | 0.6763 | 0.6550 | +0.0744 | **+0.0531** | -0.0213 |
| **Random Forest** | 0.7107 | 0.7420 | 0.7471 | +0.0313 | **+0.0364** | +0.0051 |
| **XGBoost** | 0.7317 | 0.7353 | 0.7574 | +0.0036 | **+0.0257** | +0.0221 |

### Key Scientific Findings & Guardrails:
1. **Recent DO History Adds Defensible Predictive Information:** For every evaluated algorithm, adding 2-hour DO history yields a decisive lift over current-only features alone (`Config C - Config A` lift: **+0.0531** for Logistic Regression, **+0.0364** for Random Forest, and **+0.0257** for XGBoost).
2. **Model-Specific Behavior on Full vs DO History:** Config C does *not* strictly dominate Config B across all models. For linear Logistic Regression, auxiliary water parameters in Config B provide a higher lift (**0.6763** vs. **0.6550**), whereas for non-linear ensembles (Random Forest and XGBoost), isolating DO history eliminates collinear sensor noise and achieves the highest PR-AUC (**0.7471** and **0.7574** respectively).
3. **No Explicit Derivative Engineered:** All historical features are discrete sensor observations ($t-15\text{m}$ to $t-120\text{m}$); no mathematical derivative or explicit 'DO velocity' feature was computed.