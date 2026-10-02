"""
ShinerAI: Input Validation Module
Validates incoming REST API payloads for operational validity, feature completeness,
and data integrity before running model inference.
"""

from datetime import datetime
import math
from typing import Any, Dict, Tuple

from backend.config import (
    CONFIG_C_FEATURES,
    LAG_ALIASES,
    LAG_MINUTES,
    OPERATIONAL_DO_THRESHOLD,
)


class ValidationError(Exception):
    """Raised when an API payload fails schema or domain validation."""

    def __init__(self, message: str, field: str = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.details = details

    def to_dict(self) -> Dict[str, Any]:
        result = {"error": "ValidationError", "message": self.message}
        if self.field:
            result["field"] = self.field
        if self.details:
            result["details"] = self.details
        return result


def _is_valid_number(val: Any) -> bool:
    """Checks whether a value is a finite float or int (and not boolean)."""
    if isinstance(val, bool):
        return False
    if not isinstance(val, (int, float)):
        return False
    return not (math.isnan(val) or math.isinf(val))


def parse_timestamp(ts_str: str) -> Tuple[datetime, float, float]:
    """
    Parses an ISO 8601 or standard datetime string and extracts hour and minute of day.
    """
    if not isinstance(ts_str, str) or not ts_str.strip():
        raise ValidationError("prediction_timestamp must be a non-empty string.", field="prediction_timestamp")

    cleaned = ts_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        # Fallback to common timestamp formats
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d %H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValidationError(
                f"Cannot parse prediction_timestamp '{ts_str}'. Expected ISO 8601 format (e.g., '2026-01-26T03:30:00').",
                field="prediction_timestamp",
            )

    hour_of_day = float(dt.hour)
    minute_of_day = float(dt.hour * 60 + dt.minute)
    return dt, hour_of_day, minute_of_day


def validate_prediction_payload(data: Any) -> Dict[str, Any]:
    """
    Validates and normalizes incoming JSON data for the /predict and /explain endpoints.

    Parameters
    ----------
    data : Any
        Incoming JSON payload parsed as a Python dict.

    Returns
    -------
    dict
        Validated dictionary containing:
        - 'pond_id': str
        - 'prediction_timestamp': str
        - 'features': dict mapping all 11 CONFIG_C_FEATURES to float values
    """
    if not isinstance(data, dict):
        raise ValidationError("Request body must be a JSON object.")

    # 1. Validate pond_id
    pond_id = data.get("pond_id")
    if not pond_id or not isinstance(pond_id, str) or not pond_id.strip():
        raise ValidationError("pond_id is required and must be a non-empty string.", field="pond_id")
    pond_id = pond_id.strip()

    # 2. Validate prediction_timestamp and derive temporal features
    ts_raw = data.get("prediction_timestamp")
    if ts_raw is None:
        raise ValidationError("prediction_timestamp is required (e.g., '2026-01-26T03:30:00').", field="prediction_timestamp")
    _, hour_of_day, minute_of_day = parse_timestamp(ts_raw)

    # 3. Validate current_do
    if "current_do" not in data:
        raise ValidationError("current_do is required.", field="current_do")
    current_do_val = data["current_do"]
    if not _is_valid_number(current_do_val):
        raise ValidationError("current_do must be a valid numeric value.", field="current_do")
    current_do_val = float(current_do_val)

    if current_do_val < 0.0:
        raise ValidationError(
            f"current_do ({current_do_val} mg/L) cannot be negative.",
            field="current_do",
        )

    # Operational rule: current_do must be >= 3.0 mg/L
    if current_do_val < OPERATIONAL_DO_THRESHOLD:
        raise ValidationError(
            f"Operational boundary condition violated: current_do is already {current_do_val:.2f} mg/L "
            f"(below the {OPERATIONAL_DO_THRESHOLD:.1f} mg/L threshold). ShinerAI operates only when current DO "
            f"is at or above {OPERATIONAL_DO_THRESHOLD:.1f} mg/L to predict low-DO risk within the next 2 hours.",
            field="current_do",
            details={
                "current_do": current_do_val,
                "minimum_operational_threshold": OPERATIONAL_DO_THRESHOLD,
                "status": "ALREADY_LOW_DO",
            },
        )

    # 4. Validate lag features
    features: Dict[str, float] = {
        "current_do": current_do_val,
        "hour_of_day": hour_of_day,
        "minute_of_day": minute_of_day,
    }

    missing_lags = []
    invalid_lags = []

    for m in LAG_MINUTES:
        canonical_key = f"do_t_minus_{m}"
        # Check canonical key or aliases (e.g., do_t-15m)
        found_val = None
        if canonical_key in data:
            found_val = data[canonical_key]
        else:
            # Check aliases
            alias_key = f"do_t-{m}m"
            if alias_key in data:
                found_val = data[alias_key]
            else:
                alias_key_alt = f"do_t_minus_{m}m"
                if alias_key_alt in data:
                    found_val = data[alias_key_alt]

        if found_val is None:
            missing_lags.append(canonical_key)
        elif not _is_valid_number(found_val):
            invalid_lags.append((canonical_key, found_val, "Must be a finite numeric value"))
        elif float(found_val) < 0.0:
            invalid_lags.append((canonical_key, found_val, "Cannot be negative"))
        else:
            features[canonical_key] = float(found_val)

    if missing_lags:
        raise ValidationError(
            f"Missing required historical lag features: {missing_lags}. "
            f"Provide all 8 lags from t-15m to t-120m (e.g. 'do_t_minus_15' or 'do_t-15m').",
            field="lags",
            details={"missing_lags": missing_lags},
        )

    if invalid_lags:
        raise ValidationError(
            f"Invalid lag features encountered: {invalid_lags}.",
            field="lags",
            details={"invalid_lags": invalid_lags},
        )

    # Optional overrides: if hour_of_day / minute_of_day were explicitly provided, verify compatibility
    if "hour_of_day" in data and _is_valid_number(data["hour_of_day"]):
        user_hour = float(data["hour_of_day"])
        if 0.0 <= user_hour <= 23.0:
            features["hour_of_day"] = user_hour
    if "minute_of_day" in data and _is_valid_number(data["minute_of_day"]):
        user_min = float(data["minute_of_day"])
        if 0.0 <= user_min <= 1439.0:
            features["minute_of_day"] = user_min

    # Ensure all 11 features are present
    for feat in CONFIG_C_FEATURES:
        if feat not in features:
            raise ValidationError(f"Internal error: missing prepared feature '{feat}'.")

    return {
        "pond_id": pond_id,
        "prediction_timestamp": str(ts_raw).strip(),
        "features": features,
    }
