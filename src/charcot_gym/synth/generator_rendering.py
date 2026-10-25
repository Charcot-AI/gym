"""Render synthetic clinical notes and FHIR resources."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date
from typing import Any

from charcot_gym.synth.lab_values import _format_value, _reference_range, _sample_value
from charcot_gym.terminology import UCUM_SYSTEM_URI

PRIVATE_UNITS = {"mg/dL": "mg/dl", "10*3/uL": "x10^3/uL"}

ABBREVIATIONS = {
    "2345-7": "Gluc",
    "15074-8": "Gluc",
    "4548-4": "HbA1c",
    "2160-0": "Cr",
    "3094-0": "BUN",
    "2951-2": "Na",
    "2823-3": "K",
    "2093-3": "TC",
    "2085-9": "HDL-C",
    "13457-7": "LDL-C",
    "2571-8": "TG",
    "718-7": "Hgb",
    "4544-3": "Hct",
    "6690-2": "WBC",
    "777-3": "PLT",
    "3016-3": "TSH",
    "1742-6": "ALT/SGPT",
    "1920-8": "AST/SGOT",
    "751-8": "ANC",
    "2039-6": "CEA",
    "2857-1": "PSA",
    "29463-7": "Wt",
}


@dataclass(frozen=True)
class _ReportContext:
    rng: random.Random
    layout: str
    current: list[dict[str, Any]]
    histories: list[dict[str, Any]]
    collection_date: date
    distractor_settings: dict[str, Any]
    calibration: dict[str, Any]
    sex: str


@dataclass(frozen=True)
class _CodingNoteContext:
    rng: random.Random
    layout: str
    scenario_id: str
    variant: str | None
    history_cycles: int | None
    current: list[dict[str, Any]]
    histories: list[dict[str, Any]]
    codes: list[str]
    config: dict[str, Any]


def _coding_expected(
    rng: random.Random,
    scenario_id: str,
    variant: str | None,
    current: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[str]:
    if scenario_id == "routine":
        routine = config["routine"]
        count = rng.randint(routine["code_count"]["min"], routine["code_count"]["max"])
        codes = rng.sample(routine["chronic_codes"], count)
        return codes or [routine["fallback_code"]]
    if scenario_id != "oncology_precycle":
        return list(config["scenario_codes"][scenario_id])

    oncology = config["oncology"]
    if variant == "clear":
        return list(oncology["clear_codes"])
    codes = list(oncology["hold_codes"])
    values = {item["entry"]["code"]: item["value"] for item in current}
    if values["751-8"] < oncology["anc_threshold"]:
        codes.append(oncology["anc_code"])
    if values["777-3"] < oncology["platelet_threshold"]:
        codes.append(oncology["platelet_code"])
    return codes


def _coding_phrase_facts(
    context: _CodingNoteContext,
) -> dict[str, str]:
    current = {item["entry"]["code"]: item for item in context.current}
    histories_by_code: dict[str, list[dict[str, Any]]] = {}
    for item in context.histories:
        histories_by_code.setdefault(item["entry"]["code"], []).append(item)

    def value(code: str, items: dict[str, dict[str, Any]]) -> str:
        item = items[code]
        return _format_value(item["value"], item["entry"]["precision"])

    def latest_history_value(code: str) -> str:
        item = max(histories_by_code[code], key=lambda candidate: candidate["date"])
        return _format_value(item["value"], item["entry"]["precision"])

    current_cycle = (context.history_cycles or 0) + 1
    facts = {
        "current_cycle": str(current_cycle),
        "previous_cycle": str(context.history_cycles or 0),
    }
    if context.scenario_id == "aki":
        facts["current_creatinine"] = value("2160-0", current)
        facts["baseline_creatinine"] = latest_history_value("2160-0")
    elif context.scenario_id == "uncontrolled_diabetes":
        facts["a1c"] = value("4548-4", current)
        facts["glucose"] = value("2345-7", current)
    elif context.scenario_id == "hypokalemia":
        facts["potassium"] = value("2823-3", current)
    elif context.scenario_id == "oncology_precycle":
        oncology = context.config["oncology"]
        facts["anc"] = value("751-8", current)
        facts["platelets"] = value("777-3", current)
        facts["anc_threshold"] = _format_value(
            oncology["anc_threshold"], current["751-8"]["entry"]["precision"]
        )
        facts["platelet_threshold"] = _format_value(
            oncology["platelet_threshold"], current["777-3"]["entry"]["precision"]
        )
    return facts


def _render_coding_note(context: _CodingNoteContext) -> str:
    rng = context.rng
    layout = context.layout
    scenario_id = context.scenario_id
    variant = context.variant
    current = context.current
    codes = context.codes
    config = context.config
    oncology = scenario_id == "oncology_precycle"
    voice = "soap_abbrev" if layout == "soap_abbrev" else layout
    style = "abbrev" if layout == "soap_abbrev" else "public"
    subjective_key = "oncology" if oncology else scenario_id
    subjective = config["subjective"][subjective_key]
    facts = _coding_phrase_facts(context)
    labs = []
    for item in current:
        entry = item["entry"]
        if layout == "soap_full":
            label = entry["display"]
        else:
            label = config["lab_abbreviations"][entry["code"]]
        unit = (
            PRIVATE_UNITS.get(entry["unit"], entry["unit"])
            if layout == "soap_abbrev"
            else entry["unit"]
        )
        labs.append(f"{label}: {_format_value(item['value'], entry['precision'])} {unit}")

    distractor_settings = config["distractors"]
    distractor_count = rng.randint(
        distractor_settings["count"]["min"], distractor_settings["count"]["max"]
    )
    distractors = rng.sample(distractor_settings["items"], distractor_count)
    phrasing = config["phrasing"]
    if scenario_id == "routine":
        routine_phrasing = phrasing["routine"]
        assessment = [
            rng.choice(routine_phrasing["diagnoses"][code])[style].format(**facts) for code in codes
        ]
        plan = rng.choice(routine_phrasing["plans"])[style]
    else:
        phrasing_key = f"oncology_{variant}" if oncology else scenario_id
        selected_phrasing = rng.choice(phrasing[phrasing_key])
        assessment = [selected_phrasing[f"assessment_{style}"].format(**facts)]
        if oncology and variant == "hold":
            if "T45.1X5A" in codes:
                assessment.append("Count suppression is due to chemotherapy.")
            if "D70.1" in codes:
                assessment.append("Chemotherapy-induced neutropenia is present.")
            if "D69.59" in codes:
                assessment.append("Chemotherapy-induced thrombocytopenia is present.")
        plan = selected_phrasing[f"plan_{style}"].format(**facts)
    objective = "; ".join(labs)
    if "E66.9" in codes:
        bmi_config = config["bmi_range"]
        bmi = _sample_value(
            rng,
            bmi_config["min"],
            bmi_config["max"],
            bmi_config["precision"],
        )
        objective += f"; BMI: {_format_value(bmi, bmi_config['precision'])} kg/m²"
    lines = [
        "Outpatient Progress Note",
        "Subjective:",
        subjective,
        "Objective:",
        objective,
        "Assessment:",
        *[f"- {label}" for label in assessment],
        "Plan:",
        plan,
        "Additional history:",
        *[item[voice] for item in distractors],
    ]
    return "\n".join(lines)


def _render_report(context: _ReportContext) -> str:
    rng = context.rng
    layout = context.layout
    current = context.current
    histories = context.histories
    collection_date = context.collection_date
    distractor_settings = context.distractor_settings
    calibration = context.calibration
    sex = context.sex
    lines = ["Synthetic Laboratory Report", f"Collection date: {collection_date.isoformat()}"]
    include_ranges = rng.random() < distractor_settings["reference_range_probability"]
    for item in current:
        entry = item["entry"]
        unit = entry["unit"]
        value = _format_value(item["value"], entry["precision"])
        reference = _reference_range(entry, sex, calibration)
        range_text = (
            f" (reference range {reference['min']}\u2013{reference['max']} {unit})"
            if include_ranges
            else ""
        )
        if layout == "table":
            lines.append(f"{entry['display']} | {value} {unit}{range_text}")
        elif layout == "key_value":
            lines.append(f"{entry['display']}: {value} {unit}{range_text}")
        elif layout == "narrative":
            lines.append(f"{entry['display']} measured {value} {unit}{range_text}.")
        else:
            private_unit = PRIVATE_UNITS.get(unit, unit)
            lines.append(f"{ABBREVIATIONS[entry['code']]} {value} {private_unit}{range_text}")
    for item in histories:
        entry = item["entry"]
        cycle_label = f"cycle {item['cycle']} " if "cycle" in item else ""
        label = ABBREVIATIONS[entry["code"]] if layout == "compact_abbrev" else entry["display"]
        unit = (
            PRIVATE_UNITS.get(entry["unit"], entry["unit"])
            if layout == "compact_abbrev"
            else entry["unit"]
        )
        lines.append(
            f"Previous result: {cycle_label}{label} "
            f"{_format_value(item['value'], entry['precision'])} "
            f"{unit} on {item['date']}."
        )
    if include_ranges:
        lines.append("Reference ranges are provided for adult results.")
    if rng.random() < distractor_settings["lab_comment_probability"]:
        lines.append(f"Lab comment: {rng.choice(distractor_settings['lab_comments'])}")
    return "\n".join(lines)


def _observation(
    patient_id: str,
    entry: dict[str, Any],
    value: float | int,
    effective: str,
    suffix: str,
) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": f"observation-{suffix}",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": entry["system"],
                    "code": entry["code"],
                    "display": entry["display"],
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": effective,
        "valueQuantity": {
            "value": value,
            "unit": entry["unit"],
            "system": UCUM_SYSTEM_URI,
            "code": entry["unit"],
        },
    }
