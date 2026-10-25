"""Draw coherent observation histories for synthetic patients."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from charcot_gym.synth.lab_values import (
    _draw_lipid_panel,
    _draw_value,
    _enforce_cbc_coupling,
    _is_abnormal_allowed,
    _reference_range,
    _round_value,
)


@dataclass(frozen=True)
class _HistoryContext:
    rng: random.Random
    lipid_rng: random.Random
    scenario: dict[str, Any]
    analytes: list[dict[str, Any]]
    collection_date: date
    calibration: dict[str, Any]
    history_cycles: int | None
    sex: str


@dataclass(frozen=True)
class _CurrentContext:
    rng: random.Random
    scenario: dict[str, Any]
    analytes: list[dict[str, Any]]
    histories: list[dict[str, Any]]
    calibration: dict[str, Any]
    variant: str | None
    marker_code: str | None
    marker_trend: str | None
    sex: str


def _collection_date(rng: random.Random, calibration: dict[str, Any]) -> date:
    start = date.fromisoformat(calibration["collection_date"]["start"])
    return start + timedelta(days=rng.randrange(calibration["collection_date"]["days"]))


def _draw_histories(context: _HistoryContext) -> list[dict[str, Any]]:
    rng = context.rng
    scenario = context.scenario
    analytes = context.analytes
    collection_date = context.collection_date
    calibration = context.calibration
    if scenario["id"] == "oncology_precycle":
        assert context.history_cycles is not None
        days_between = scenario["history_cycles"]["days_between"]
        result = []
        for cycle in range(1, context.history_cycles + 1):
            current_cycle = context.history_cycles + 1
            effective = (
                collection_date - timedelta(days=days_between * (current_cycle - cycle))
            ).isoformat()
            cycle_result = []
            for entry in analytes:
                value = _draw_value(
                    rng,
                    entry,
                    scenario["history_rules"].get(entry["code"]),
                    calibration,
                    context.sex,
                    _is_abnormal_allowed(entry, scenario, calibration),
                )
                cycle_result.append(
                    {
                        "entry": entry,
                        "value": value,
                        "date": effective,
                        "cycle": cycle,
                    }
                )
            _enforce_cbc_coupling(cycle_result, rng, calibration, scenario, context.sex)
            result.extend(cycle_result)
        return result
    settings = calibration["history_results"]
    lipid_config = calibration["lipid_panel"]
    lipid_codes = {
        lipid_config["total_cholesterol_code"],
        lipid_config["hdl_code"],
        lipid_config["ldl_code"],
        lipid_config["triglycerides_code"],
    }
    lipid_entries = [entry for entry in analytes if entry["code"] in lipid_codes]
    prior_lipid_values: dict[str, float | int] = {}
    prior_lipid_date: str | None = None
    if lipid_codes.issubset({entry["code"] for entry in lipid_entries}) and (
        context.lipid_rng.random() < lipid_config["prior_history_probability"]
    ):
        prior_lipid_values = _draw_lipid_panel(
            context.lipid_rng,
            lipid_entries,
            scenario["history_rules"],
            calibration,
            context.sex,
            scenario,
        )
        days_before = context.lipid_rng.randint(
            settings["min_days_before"], settings["max_days_before"]
        )
        prior_lipid_date = (collection_date - timedelta(days=days_before)).isoformat()
    required_codes = [
        code for code in scenario["history_rules"] if any(item["code"] == code for item in analytes)
    ]
    count = min(
        max(rng.randint(settings["min"], settings["max"]), len(required_codes)),
        settings["max"],
    )
    entries = _select_history_entries(rng, analytes, required_codes, count)
    result = []
    if prior_lipid_date is not None:
        result.extend(
            {
                "entry": entry,
                "value": prior_lipid_values[entry["code"]],
                "date": prior_lipid_date,
            }
            for entry in lipid_entries
        )
    paired_codes = {"718-7", "4544-3"}
    lipid_values = _draw_lipid_panel(
        rng,
        entries,
        scenario["history_rules"],
        calibration,
        context.sex,
        scenario,
    )
    lipid_codes = set(lipid_values)
    paired_date = None
    lipid_date = None
    if paired_codes.issubset({entry["code"] for entry in entries}):
        days_before = rng.randint(settings["min_days_before"], settings["max_days_before"])
        paired_date = (collection_date - timedelta(days=days_before)).isoformat()
    if lipid_values:
        days_before = rng.randint(settings["min_days_before"], settings["max_days_before"])
        lipid_date = (collection_date - timedelta(days=days_before)).isoformat()
    for entry in entries:
        value_rule = scenario["history_rules"].get(entry["code"])
        value = (
            lipid_values[entry["code"]]
            if entry["code"] in lipid_values
            else _draw_value(
                rng,
                entry,
                value_rule,
                calibration,
                context.sex,
                _is_abnormal_allowed(entry, scenario, calibration),
            )
        )
        if entry["code"] in lipid_codes and lipid_date is not None:
            effective = lipid_date
        elif entry["code"] in paired_codes and paired_date is not None:
            effective = paired_date
        else:
            days_before = rng.randint(settings["min_days_before"], settings["max_days_before"])
            effective = (collection_date - timedelta(days=days_before)).isoformat()
        result.append(
            {
                "entry": entry,
                "value": value,
                "date": effective,
            }
        )
    _enforce_cbc_coupling(result, rng, calibration, scenario, context.sex)
    return result


def _select_history_entries(
    rng: random.Random,
    analytes: list[dict[str, Any]],
    required_codes: list[str],
    count: int,
) -> list[dict[str, Any]]:
    by_code = {entry["code"]: entry for entry in analytes}
    paired_codes = {"718-7", "4544-3"}
    entries = [by_code[code] for code in required_codes]
    if paired_codes.intersection(required_codes):
        entries.extend(
            by_code[code] for code in paired_codes - set(required_codes) if code in by_code
        )
    count = max(count, len(entries))
    remaining = [
        entry
        for entry in analytes
        if entry["code"] not in required_codes and entry["code"] not in paired_codes
    ]
    groups = [[entry] for entry in remaining]
    if paired_codes.issubset(by_code) and not paired_codes.intersection(required_codes):
        groups.append([by_code["718-7"], by_code["4544-3"]])
    rng.shuffle(groups)
    for group in groups:
        if len(entries) + len(group) <= count:
            entries.extend(group)
    return entries


def _draw_current(context: _CurrentContext) -> list[dict[str, Any]]:
    rng = context.rng
    scenario = context.scenario
    analytes = context.analytes
    histories = context.histories
    calibration = context.calibration
    variant = context.variant
    marker_code = context.marker_code
    marker_trend = context.marker_trend
    sex = context.sex
    lipid_values = _draw_lipid_panel(
        rng,
        analytes,
        scenario["current_rules"],
        calibration,
        sex,
        scenario,
    )
    result = []
    for entry in analytes:
        rule = scenario["current_rules"].get(entry["code"])
        if entry["code"] in lipid_values:
            value = lipid_values[entry["code"]]
        elif scenario["id"] == "oncology_precycle" and entry["code"] in {
            "751-8",
            "777-3",
        }:
            assert variant is not None
            assert rule is not None
            value = _draw_value(
                rng,
                entry,
                rule[variant],
                calibration,
                sex,
                _is_abnormal_allowed(entry, scenario, calibration),
            )
        elif scenario["id"] == "oncology_precycle" and entry["code"] == marker_code:
            assert marker_code is not None
            assert marker_trend is not None
            history = max(
                (item for item in histories if item["entry"]["code"] == marker_code),
                key=lambda item: item["date"],
            )
            trend_rule = scenario["current_rules"]["tumor_marker"]
            value = _draw_marker_trend(
                rng,
                history["value"],
                marker_trend,
                trend_rule,
                entry,
                calibration,
                sex,
            )
        elif rule and "min_ratio_to_history" in rule:
            history_by_code = {item["entry"]["code"]: item["value"] for item in histories}
            minimum = max(
                rule["min"],
                history_by_code[entry["code"]] * rule["min_ratio_to_history"]
                + 10 ** (-entry["precision"]),
            )
            value = _draw_value(
                rng,
                entry,
                {**rule, "min": minimum},
                calibration,
                sex,
                _is_abnormal_allowed(entry, scenario, calibration),
            )
        else:
            value = _draw_value(
                rng,
                entry,
                rule,
                calibration,
                sex,
                _is_abnormal_allowed(entry, scenario, calibration),
            )
        result.append({"entry": entry, "value": value})
    _enforce_cbc_coupling(result, rng, calibration, scenario, sex)
    return result


def _draw_marker_trend(  # noqa: PLR0913, PLR0917
    rng: random.Random,
    prior_value: float | int,
    trend: str,
    rule: dict[str, Any],
    entry: dict[str, Any],
    calibration: dict[str, Any],
    sex: str,
) -> float | int:
    reference = _reference_range(entry, sex, calibration)
    parameters = calibration["analyte_parameters"][entry["code"]]
    lower = max(reference["min"], parameters["physiologic_min"])
    upper = min(reference["max"], parameters["physiologic_max"])
    minimum_delta = rule["delta_min"]
    maximum_delta = rule["delta_max"]
    directions = [trend, "falling" if trend == "rising" else "rising"]
    for direction in directions:
        if direction == "rising":
            maximum = min(maximum_delta, upper - prior_value)
        else:
            maximum = min(maximum_delta, prior_value - lower)
        if maximum < minimum_delta:
            continue
        for _ in range(10_000):
            delta = rng.uniform(minimum_delta, maximum)
            value = _round_value(
                prior_value + delta if direction == "rising" else prior_value - delta,
                entry["precision"],
            )
            if lower <= value <= upper:
                return value
    raise ValueError(f"No valid tumor-marker trend for {entry['code']} at {prior_value}.")


def _validate_generated_values(
    values: list[dict[str, Any]],
    scenario: dict[str, Any],
    sex: str,
    calibration: dict[str, Any],
) -> None:
    for item in values:
        entry = item["entry"]
        value = item["value"]
        parameters = calibration["analyte_parameters"][entry["code"]]
        precision = parameters["precision"]
        if (
            value <= 0
            or value < parameters["physiologic_min"]
            or value > parameters["physiologic_max"]
            or value != _round_value(value, precision)
        ):
            raise ValueError(f"Invalid generated value for {entry['code']}: {value}.")
        if not _is_abnormal_allowed(entry, scenario, calibration):
            reference = _reference_range(entry, sex, calibration)
            if not reference["min"] <= value <= reference["max"]:
                raise ValueError(f"Unexpected out-of-range value for {entry['code']}: {value}.")


def _apply_history_relationships(  # noqa: PLR0913, PLR0917
    rng: random.Random,
    scenario: dict[str, Any],
    histories: list[dict[str, Any]],
    current: list[dict[str, Any]],
    sex: str,
    calibration: dict[str, Any],
) -> None:
    relationships = scenario.get("history_relationships", {})
    if not relationships:
        return

    current_by_code = {item["entry"]["code"]: item for item in current}
    for code, relationship in relationships.items():
        if relationship["direction"] != "higher_than_following":
            raise ValueError(f"Unsupported history relationship: {relationship['direction']}")
        current_item = current_by_code[code]
        precision = current_item["entry"]["precision"]
        scale = 10**precision
        minimum_step = round(relationship["min_delta"] * scale)
        maximum_step = round(relationship["max_delta"] * scale)
        following_value = current_item["value"]
        prior_items = sorted(
            (item for item in histories if item["entry"]["code"] == code),
            key=lambda item: item["cycle"],
            reverse=True,
        )
        for item in prior_items:
            delta = rng.randint(minimum_step, maximum_step) / scale
            item["value"] = round(following_value + delta, precision)
            following_value = item["value"]

    history_groups: dict[int, list[dict[str, Any]]] = {}
    for item in histories:
        history_groups.setdefault(item["cycle"], []).append(item)
    for group in history_groups.values():
        _enforce_cbc_coupling(group, rng, calibration, scenario, sex)
