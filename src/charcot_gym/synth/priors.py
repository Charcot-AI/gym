"""Validate priors and sample calibrated laboratory and history data."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)
PERCENTILE_KEYS = ("p05", "p25", "p50", "p75", "p95")


def load_priors(path: Path) -> dict[str, Any] | None:
    """Load a valid priors file when it exists."""
    if not path.is_file():
        return None
    return validate_priors(json.loads(path.read_text(encoding="utf-8")))


def validate_priors(value: Any) -> dict[str, Any]:
    """Validate the fields required by Hard task generation."""
    if not isinstance(value, Mapping):
        raise ValueError("priors must be a JSON object")
    data = dict(value)
    _validate_weighted_rows(data.get("condition_categories"), "icd10_category", "patient_share")
    _validate_weighted_rows(data.get("lab_frequency"), "loinc", "share")
    histogram = data.get("conditions_per_patient")
    expected_counts = {*(str(count) for count in range(1, 10)), "10+"}
    if not isinstance(histogram, Mapping) or set(histogram) != expected_counts:
        raise ValueError("conditions_per_patient must define counts 1 through 9 and 10+")
    _validate_shares(histogram.values(), "conditions_per_patient")
    oncology_share = data.get("oncology_patient_share")
    if not isinstance(oncology_share, (int, float)) or isinstance(oncology_share, bool):
        raise ValueError("oncology_patient_share must be between 0 and 1")
    if not math.isfinite(oncology_share) or not 0 <= oncology_share <= 1:
        raise ValueError("oncology_patient_share must be between 0 and 1")
    percentiles = data.get("lab_value_percentiles")
    if not isinstance(percentiles, list):
        raise ValueError("lab_value_percentiles must be a list")
    for row in percentiles:
        if not isinstance(row, Mapping) or not isinstance(row.get("loinc"), str):
            raise ValueError("lab_value_percentiles rows need a LOINC code")
        if not isinstance(row.get("ucum"), str):
            raise ValueError("lab_value_percentiles rows need a UCUM unit")
        values = [row.get(key) for key in PERCENTILE_KEYS]
        if not all(_is_number(item) for item in values):
            raise ValueError("lab_value_percentiles values must be finite numbers")
        numeric_values = [float(row[key]) for key in PERCENTILE_KEYS]
        if numeric_values != sorted(numeric_values):
            raise ValueError("lab_value_percentiles must be nondecreasing")
    if not isinstance(data.get("lab_pairs_same_encounter"), list):
        raise ValueError("lab_pairs_same_encounter must be a list")
    return data


def panel_weights(
    defaults: Mapping[str, float],
    analytes: list[dict[str, Any]],
    priors: Mapping[str, Any],
) -> dict[str, float]:
    """Calculate normalized panel weights from observed lab shares."""
    shares = {row["loinc"]: row["share"] for row in priors["lab_frequency"]}
    weights = {
        panel: sum(
            shares.get(entry["code"], 0.0) for entry in analytes if entry.get("panel") == panel
        )
        for panel in defaults
    }
    present_total = sum(weight for weight in weights.values() if weight > 0)
    missing_panels = [panel for panel, weight in weights.items() if weight == 0]
    missing_default_total = sum(defaults[panel] for panel in missing_panels)
    if present_total == 0:
        weights = dict(defaults)
    elif missing_default_total > 0:
        for panel in missing_panels:
            weights[panel] = defaults[panel] * present_total / missing_default_total
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("panel weights must have a positive total")
    return {panel: weight / total for panel, weight in weights.items()}


def choose_panels(
    rng: random.Random,
    analytes: list[dict[str, Any]],
    calibration: Mapping[str, Any],
    priors: Mapping[str, Any],
) -> list[str]:
    """Choose report panels without replacement using calibrated weights."""
    weights = panel_weights(calibration["panel_weights"], analytes, priors)
    count = rng.randint(
        calibration["panels_per_report"]["min"],
        calibration["panels_per_report"]["max"],
    )
    selected = []
    for _ in range(count):
        available = [panel for panel in weights if panel not in selected]
        selected.append(
            rng.choices(available, weights=[weights[panel] for panel in available], k=1)[0]
        )
    return selected


def matching_percentiles(
    priors: Mapping[str, Any], loinc: str, ucum: str
) -> dict[str, float] | None:
    """Return percentiles only when both the LOINC code and unit match."""
    for row in priors["lab_value_percentiles"]:
        if row["loinc"] == loinc and row["ucum"] == ucum:
            return {key: float(row[key]) for key in PERCENTILE_KEYS}
    return None


def sample_prior_value(
    rng: random.Random,
    percentiles: Mapping[str, float],
    lower: float,
    upper: float,
    precision: int,
) -> float | int:
    """Sample one value from the piecewise-linear percentile distribution."""
    quantile = rng.uniform(QUANTILES[0], QUANTILES[-1])
    segment = next(
        index
        for index in range(len(QUANTILES) - 1)
        if QUANTILES[index] <= quantile <= QUANTILES[index + 1]
    )
    start = float(percentiles[PERCENTILE_KEYS[segment]])
    end = float(percentiles[PERCENTILE_KEYS[segment + 1]])
    fraction = (quantile - QUANTILES[segment]) / (QUANTILES[segment + 1] - QUANTILES[segment])
    value = min(max(start + (end - start) * fraction, lower), upper)
    scale = 10**precision
    minimum = math.ceil(lower * scale - 1e-9) / scale
    maximum = math.floor(upper * scale + 1e-9) / scale
    if minimum > maximum:
        raise ValueError("physiologic interval has no value at the configured precision")
    rounded = round(value, precision)
    rounded = min(max(rounded, minimum), maximum)
    return int(rounded) if precision == 0 else rounded


def sample_condition_count(rng: random.Random, priors: Mapping[str, Any]) -> int:
    """Sample a history count from the supplied distribution."""
    labels = [str(count) for count in range(1, 10)] + ["10+"]
    label = rng.choices(
        labels,
        weights=[priors["conditions_per_patient"][item] for item in labels],
        k=1,
    )[0]
    return rng.randint(10, 14) if label == "10+" else int(label)


def _validate_weighted_rows(value: Any, key: str, weight_key: str) -> None:
    if not isinstance(value, list):
        raise ValueError(f"{key} entries must be a list")
    for row in value:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get(key), str)
            or not _is_number(row.get(weight_key))
            or row[weight_key] < 0
        ):
            raise ValueError(f"{key} entries need a string key and nonnegative weight")


def _validate_shares(values: Any, label: str) -> None:
    if not all(_is_number(value) and value >= 0 for value in values):
        raise ValueError(f"{label} values must be finite and nonnegative")
    if not 0.99 <= sum(values) <= 1.01:
        raise ValueError(f"{label} values must sum to 1")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
