"""Shared physiological lab value sampling and coupling helpers."""

from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any, cast

from charcot_gym.synth.priors import matching_percentiles, sample_prior_value


def _draw_value(  # noqa: PLR0913, PLR0917
    rng: random.Random,
    entry: dict[str, Any],
    rule: dict[str, Any] | None,
    calibration: dict[str, Any],
    sex: str,
    allow_abnormal: bool,
    priors: Mapping[str, Any] | None = None,
) -> float | int:
    if priors is not None and not allow_abnormal:
        percentiles = matching_percentiles(priors, entry["code"], entry["unit"])
        if percentiles is not None:
            parameters = calibration["analyte_parameters"][entry["code"]]
            lower = parameters["physiologic_min"]
            upper = parameters["physiologic_max"]
            if rule is not None:
                range_rule = rule.get("by_sex", {}).get(sex, rule)
                lower = max(lower, range_rule.get("min", lower))
                upper = min(upper, range_rule.get("max", upper))
            return sample_prior_value(rng, percentiles, lower, upper, entry["precision"])
    lower, upper = _value_bounds(entry, rule, calibration, sex, allow_abnormal)
    return _sample_value(rng, lower, upper, entry["precision"])


def _value_bounds(
    entry: dict[str, Any],
    rule: dict[str, Any] | None,
    calibration: dict[str, Any],
    sex: str,
    allow_abnormal: bool,
) -> tuple[float, float]:
    parameters = calibration["analyte_parameters"][entry["code"]]
    lower = parameters["physiologic_min"]
    upper = parameters["physiologic_max"]
    if rule is not None:
        range_rule = rule.get("by_sex", {}).get(sex, rule)
        lower = max(lower, range_rule["min"])
        upper = min(upper, range_rule["max"])
    if not allow_abnormal:
        reference = _reference_range(entry, sex, calibration)
        lower = max(lower, reference["min"])
        upper = min(upper, reference["max"])
    return lower, upper


def _draw_lipid_panel(  # noqa: PLR0913, PLR0917
    rng: random.Random,
    entries: list[dict[str, Any]],
    rules: dict[str, Any],
    calibration: dict[str, Any],
    sex: str,
    scenario: dict[str, Any],
    priors: Mapping[str, Any] | None = None,
) -> dict[str, float | int]:
    config = calibration["lipid_panel"]
    codes = {
        "total": config["total_cholesterol_code"],
        "hdl": config["hdl_code"],
        "ldl": config["ldl_code"],
        "triglycerides": config["triglycerides_code"],
    }
    by_code = {entry["code"]: entry for entry in entries}
    if not set(codes.values()).issubset(by_code):
        return {}

    triglycerides_entry = by_code[codes["triglycerides"]]
    tg_step = 10 ** (-triglycerides_entry["precision"])
    tg_upper = config["triglycerides_max_exclusive"] - tg_step
    triglycerides_rule = rules.get(codes["triglycerides"])
    if triglycerides_rule is not None:
        triglycerides_rule = triglycerides_rule.get("by_sex", {}).get(sex, triglycerides_rule)
    triglycerides_rule = {
        **(triglycerides_rule or {}),
        "min": (triglycerides_rule or {}).get(
            "min",
            calibration["analyte_parameters"][codes["triglycerides"]]["physiologic_min"],
        ),
        "max": min((triglycerides_rule or {}).get("max", tg_upper), tg_upper),
    }

    total_entry = by_code[codes["total"]]
    total_percentiles = (
        matching_percentiles(priors, total_entry["code"], total_entry["unit"])
        if priors is not None
        else None
    )
    if total_percentiles is not None:
        parameters = calibration["analyte_parameters"][total_entry["code"]]
        total_bounds = (
            max(parameters["physiologic_min"], total_percentiles["p05"]),
            min(parameters["physiologic_max"], total_percentiles["p95"]),
        )
    else:
        total_bounds = _value_bounds(
            total_entry,
            rules.get(codes["total"]),
            calibration,
            sex,
            _is_abnormal_allowed(total_entry, scenario, calibration),
        )
    for _ in range(10_000):
        values = {
            codes["hdl"]: _draw_value(
                rng,
                by_code[codes["hdl"]],
                rules.get(codes["hdl"]),
                calibration,
                sex,
                _is_abnormal_allowed(by_code[codes["hdl"]], scenario, calibration),
                priors,
            ),
            codes["triglycerides"]: _draw_value(
                rng,
                triglycerides_entry,
                triglycerides_rule,
                calibration,
                sex,
                _is_abnormal_allowed(triglycerides_entry, scenario, calibration),
                priors,
            ),
            codes["ldl"]: _draw_value(
                rng,
                by_code[codes["ldl"]],
                rules.get(codes["ldl"]),
                calibration,
                sex,
                _is_abnormal_allowed(by_code[codes["ldl"]], scenario, calibration),
                priors,
            ),
        }
        total = (
            values[codes["ldl"]] + values[codes["hdl"]] + round(values[codes["triglycerides"]] / 5)
        )
        if total_bounds[0] <= total <= total_bounds[1]:
            values[codes["total"]] = total
            return values
    raise ValueError("Could not sample a coherent lipid panel within configured bounds.")


def _sample_value(rng: random.Random, lower: float, upper: float, precision: int) -> float | int:
    if lower > upper:
        raise ValueError(f"No valid value in configured interval [{lower}, {upper}].")
    for _ in range(10_000):
        value = _round_value(rng.uniform(lower, upper), precision)
        if value > 0 and lower <= value <= upper:
            return value
    raise ValueError(f"Could not sample a positive value in [{lower}, {upper}].")


def _is_abnormal_allowed(
    entry: dict[str, Any], scenario: dict[str, Any], calibration: dict[str, Any]
) -> bool:
    parameters = calibration["analyte_parameters"].get(entry["code"], {})
    return parameters.get("name") in scenario.get("abnormal_analytes", [])


def _reference_range(
    entry: dict[str, Any], sex: str, calibration: dict[str, Any]
) -> dict[str, float]:
    return cast(
        dict[str, float],
        calibration.get("sex_specific_reference_ranges", {})
        .get(entry["code"], {})
        .get(sex, entry["reference_range"]),
    )


def _enforce_cbc_coupling(  # noqa: PLR0913, PLR0917
    values: list[dict[str, Any]],
    rng: random.Random,
    calibration: dict[str, Any],
    scenario: dict[str, Any],
    sex: str,
    priors: Mapping[str, Any] | None = None,
) -> None:
    by_code = {item["entry"]["code"]: item for item in values}
    abnormal_analytes = set(scenario.get("abnormal_analytes", []))
    if "718-7" in by_code and "4544-3" in by_code:
        hgb_item = by_code["718-7"]
        hgb_entry = hgb_item["entry"]
        hgb_parameters = calibration["analyte_parameters"][hgb_entry["code"]]
        hct_item = by_code["4544-3"]
        hct_entry = hct_item["entry"]
        hct_parameters = calibration["analyte_parameters"][hct_entry["code"]]
        factor_rule = calibration["hematology_coupling"]["hematocrit_factor"]
        hgb_range = _coupling_range(hgb_entry, hgb_parameters, scenario, sex, calibration, priors)
        hct_range = _coupling_range(hct_entry, hct_parameters, scenario, sex, calibration, priors)
        hgb_min = max(hgb_range["min"], hgb_parameters["physiologic_min"])
        hgb_max = min(hgb_range["max"], hgb_parameters["physiologic_max"])
        hct_min = max(hct_range["min"], hct_parameters["physiologic_min"])
        hct_max = min(hct_range["max"], hct_parameters["physiologic_max"])
        feasible_hgb_min = max(hgb_min, hct_min / factor_rule["max"])
        feasible_hgb_max = min(hgb_max, hct_max / factor_rule["min"])
        if feasible_hgb_min > feasible_hgb_max:
            raise ValueError("No valid coupled hemoglobin and hematocrit interval.")
        if not feasible_hgb_min <= hgb_item["value"] <= feasible_hgb_max:
            percentiles = (
                matching_percentiles(priors, hgb_entry["code"], hgb_entry["unit"])
                if priors is not None and not _is_abnormal_allowed(hgb_entry, scenario, calibration)
                else None
            )
            if percentiles is None:
                hgb_item["value"] = _sample_value(
                    rng, feasible_hgb_min, feasible_hgb_max, hgb_entry["precision"]
                )
            else:
                hgb_item["value"] = sample_prior_value(
                    rng, percentiles, feasible_hgb_min, feasible_hgb_max, hgb_entry["precision"]
                )
        factor_min = max(factor_rule["min"], hct_min / hgb_item["value"])
        factor_max = min(factor_rule["max"], hct_max / hgb_item["value"])
        for _ in range(10_000):
            factor = rng.uniform(factor_min, factor_max)
            hct_value = _round_value(hgb_item["value"] * factor, hct_entry["precision"])
            if hct_min <= hct_value <= hct_max:
                hct_item["value"] = hct_value
                break
        else:
            raise ValueError("Could not sample coupled hematocrit within valid bounds.")
    if "751-8" not in by_code or "6690-2" not in by_code:
        return
    if not {"ANC", "WBC"}.issubset(abnormal_analytes):
        return
    anc = by_code["751-8"]["value"]
    wbc_item = by_code["6690-2"]
    wbc_entry = wbc_item["entry"]
    fraction_rule = calibration["hematology_coupling"]["neutrophil_fraction"]
    fraction = rng.uniform(fraction_rule["min"], fraction_rule["max"])
    wbc_item["value"] = _round_value(anc / fraction, wbc_entry["precision"])


def _coupling_range(  # noqa: PLR0913, PLR0917
    entry: dict[str, Any],
    parameters: dict[str, Any],
    scenario: dict[str, Any],
    sex: str,
    calibration: dict[str, Any],
    priors: Mapping[str, Any] | None,
) -> dict[str, float]:
    if _is_abnormal_allowed(entry, scenario, calibration):
        return {
            "min": parameters["physiologic_min"],
            "max": parameters["physiologic_max"],
        }
    if priors is not None:
        percentiles = matching_percentiles(priors, entry["code"], entry["unit"])
        if percentiles is not None:
            return {
                "min": max(parameters["physiologic_min"], percentiles["p05"]),
                "max": min(parameters["physiologic_max"], percentiles["p95"]),
            }
    return _reference_range(entry, sex, calibration)


def _format_value(value: float | int, precision: int) -> str:
    return f"{value:.{precision}f}"


def _round_value(value: float, precision: int) -> float | int:
    rounded = round(value, precision)
    return int(rounded) if precision == 0 else rounded
