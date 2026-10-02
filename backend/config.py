"""
ShinerAI: Backend Configuration
Manages artifact paths, feature schemas, and runtime operational parameters.
"""

import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Default model artifact path (Config C XGBoost)
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "xgboost_config_c.joblib"

# Configurable model artifact path via environment variable
MODEL_ARTIFACT_PATH = os.environ.get("MODEL_ARTIFACT_PATH", str(DEFAULT_MODEL_PATH))

# Model metadata path
METADATA_PATH = BASE_DIR / "models" / "model_metadata.json"

# Decision threshold for binary classification (AT_RISK vs SAFE)
DECISION_THRESHOLD = float(os.environ.get("DECISION_THRESHOLD", "0.50"))

# Operational dissolved oxygen threshold (mg/L)
# ShinerAI operates on pond states where current DO >= 3.0 mg/L to predict low-DO events
OPERATIONAL_DO_THRESHOLD = 3.0

# Prediction horizon in hours
PREDICTION_HORIZON_HOURS = 2.0

# Exact ordered features for Config C tree models
CONFIG_C_FEATURES = [
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
    "do_t_minus_120",
]

# Supported lag minute offsets
LAG_MINUTES = [15, 30, 45, 60, 75, 90, 105, 120]

# Mapping from alternative client key names to canonical feature names
LAG_ALIASES = {
    f"do_t-{m}m": f"do_t_minus_{m}" for m in LAG_MINUTES
}
LAG_ALIASES.update({
    f"do_t_minus_{m}m": f"do_t_minus_{m}" for m in LAG_MINUTES
})
