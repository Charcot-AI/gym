"""Format synthetic laboratory reports and their result rows."""

from __future__ import annotations

import hashlib
from datetime import date
from typing import Any

HEMOGLOBIN_ALTERNATIVE_UNIT = "g/L"

REPORT_LAYOUTS = ("reference_table", "portal_summary", "fax_printout")
HEMOGLOBIN_SEX_AGE_RANGES = {
    ("female", True): {"min": 12.0, "max": 16.0},
    ("male", True): {"min": 13.0, "max": 17.0},
    ("female", False): {"min": 11.5, "max": 15.5},
    ("male", False): {"min": 12.5, "max": 16.5},
}


def _sex_age_reference_table(unit: str) -> str:
    if unit not in {"g/dL", "g/L"}:
        raise ValueError(f"unsupported hemoglobin display unit: {unit}")
    scale = 10 if unit == "g/L" else 1

    def format_bound(value: float) -> str:
        return f"{value * scale:.0f}" if scale == 10 else f"{value:.1f}"

    intervals = (
        ("female", "<65", HEMOGLOBIN_SEX_AGE_RANGES[("female", True)]),
        ("male", "<65", HEMOGLOBIN_SEX_AGE_RANGES[("male", True)]),
        ("female", ">=65", HEMOGLOBIN_SEX_AGE_RANGES[("female", False)]),
        ("male", ">=65", HEMOGLOBIN_SEX_AGE_RANGES[("male", False)]),
    )
    formatted = "; ".join(
        f"{sex} age {age} years {format_bound(bounds['min'])}-{format_bound(bounds['max'])}"
        for sex, age, bounds in intervals
    )
    return f"Hemoglobin ({unit}): {formatted}\n"


def _latest_observations(resources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations = [
        resource for resource in resources if resource.get("resourceType") == "Observation"
    ]
    latest_date = max(resource["effectiveDateTime"] for resource in observations)
    return sorted(
        (resource for resource in observations if resource["effectiveDateTime"] == latest_date),
        key=lambda resource: resource["code"]["coding"][0]["code"],
    )


def _row_from_observation(
    observation: dict[str, Any],
    entries_by_code: dict[str, dict[str, Any]],
    effective_date: date,
    reference_ranges: dict[str, dict[str, float]],
) -> dict[str, Any]:
    quantity = observation["valueQuantity"]
    code = observation["code"]["coding"][0]["code"]
    entry = entries_by_code[code]
    return {
        "code": code,
        "display": entry["display"],
        "value": quantity["value"],
        "unit": quantity["code"],
        "precision": entry["precision"],
        "reference_range": reference_ranges[code],
        "effective": effective_date.isoformat(),
    }


def _convert_hemoglobin_to_g_per_l(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        if row["code"] == "718-7":
            row["value"] = round(float(row["value"]) * 10)
            row["unit"] = HEMOGLOBIN_ALTERNATIVE_UNIT
            row["precision"] = 0
            row["reference_range"] = {
                key: value * 10 for key, value in row["reference_range"].items()
            }
            return
    raise ValueError("hemoglobin is required for the alternative-unit trap")


def _corrected_rows(
    rows: list[dict[str, Any]],
    calibration: dict[str, Any],
    *,
    enabled: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    corrected = [dict(row) for row in rows]
    if not enabled:
        return corrected, []
    by_code = {row["code"]: row for row in corrected}
    changed = []
    code = "777-3"
    row = by_code.get(code)
    if row is not None:
        parameters = calibration["analyte_parameters"][code]
        precision = int(parameters["precision"])
        step = 10 ** (-precision)
        current = float(row["value"])
        candidate = _rounded_value(current + step, precision)
        if candidate > parameters["physiologic_max"] or candidate == row["value"]:
            candidate = _rounded_value(current - step, precision)
        if parameters["physiologic_min"] <= candidate <= parameters["physiologic_max"]:
            row["value"] = candidate
            if row.get("display_unit") == "/µL":
                row["display_value"] = f"{round(float(candidate) * 1000):,}"
            changed.append(code)
    if not changed:
        raise ValueError("no physiologically valid correction candidate was available")
    return corrected, changed


def _prior_rows(
    latest_observations: list[dict[str, Any]],
    entries_by_code: dict[str, dict[str, Any]],
    reference_ranges: dict[str, dict[str, float]],
) -> tuple[list[dict[str, Any]], date]:
    prior_date = date.fromisoformat(latest_observations[0]["effectiveDateTime"])
    prior_rows = [
        {
            "code": observation["code"]["coding"][0]["code"],
            "display": entries_by_code[observation["code"]["coding"][0]["code"]]["display"],
            "value": observation["valueQuantity"]["value"],
            "unit": observation["valueQuantity"]["code"],
            "precision": entries_by_code[observation["code"]["coding"][0]["code"]]["precision"],
            "reference_range": reference_ranges[observation["code"]["coding"][0]["code"]],
            "effective": observation["effectiveDateTime"],
        }
        for observation in latest_observations[:3]
    ]
    return prior_rows, prior_date


def _foreign_patient_rows(
    rows: list[dict[str, Any]], calibration: dict[str, Any]
) -> list[dict[str, Any]]:
    foreign_rows = [dict(row) for row in rows]
    by_code = {row["code"]: row for row in foreign_rows}
    lipid_config = calibration["lipid_panel"]
    ldl_code = lipid_config["ldl_code"]
    hdl_code = lipid_config["hdl_code"]
    triglycerides_code = lipid_config["triglycerides_code"]
    total_code = lipid_config["total_cholesterol_code"]
    if not {ldl_code, hdl_code, triglycerides_code, total_code} <= set(by_code):
        raise ValueError("wrong-patient report must contain the complete lipid panel")
    _shift_foreign_value(by_code[ldl_code], calibration, 17)
    _shift_foreign_value(by_code[hdl_code], calibration, 7)
    _shift_foreign_value(by_code[triglycerides_code], calibration, 10)
    total = round(
        float(by_code[ldl_code]["value"])
        + float(by_code[hdl_code]["value"])
        + round(float(by_code[triglycerides_code]["value"]) / 5)
    )
    total_parameters = calibration["analyte_parameters"][total_code]
    if total < total_parameters["physiologic_min"] or total > total_parameters["physiologic_max"]:
        raise ValueError("foreign-patient lipid panel is outside physiologic bounds")
    by_code[total_code]["value"] = total
    return foreign_rows


def _shift_foreign_value(
    row: dict[str, Any], calibration: dict[str, Any], requested_delta: float
) -> None:
    parameters = calibration["analyte_parameters"][row["code"]]
    precision = row["precision"]
    step = 10 ** (-precision)
    current = float(row["value"])
    delta = requested_delta
    if row["code"] == calibration["lipid_panel"]["triglycerides_code"]:
        delta = min(
            requested_delta,
            calibration["lipid_panel"]["triglycerides_max_exclusive"] - step - current,
        )
    if delta <= 0:
        delta = -requested_delta
    candidate = _rounded_value(current + delta, precision)
    if candidate > parameters["physiologic_max"] or candidate < parameters["physiologic_min"]:
        candidate = _rounded_value(current - requested_delta, precision)
    if candidate < parameters["physiologic_min"] or candidate > parameters["physiologic_max"]:
        raise ValueError(f"could not create foreign-patient value for {row['code']}")
    row["value"] = candidate


def _range_annotation(row: dict[str, Any]) -> tuple[str, str]:
    if "value_code" in row:
        return "—", "—"
    reference_range = row["reference_range"]
    display_range = row.get("display_reference_range", reference_range)
    display_precision = row.get("display_precision", row["precision"])
    grouped_reference = "display_reference_range" in row
    minimum = reference_range.get("min")
    maximum = reference_range.get("max")
    value = float(row["value"])
    if minimum is not None and value < minimum:
        flag = "L"
    elif maximum is not None and value > maximum:
        flag = "H"
    else:
        flag = "—"
    display_minimum = display_range.get("min")
    display_maximum = display_range.get("max")
    if display_minimum is not None and display_maximum is not None:
        interval = (
            f"{_format_value(display_minimum, display_precision, grouping=grouped_reference)}\u2013"
            f"{_format_value(display_maximum, display_precision, grouping=grouped_reference)}"
        )
    elif display_minimum is not None:
        lower = (
            f"{display_minimum:g}"
            if row["code"] == "98979-8"
            else _format_value(display_minimum, display_precision, grouping=grouped_reference)
        )
        interval = f">={lower}"
    elif display_maximum is not None:
        interval = (
            f"<={_format_value(display_maximum, display_precision, grouping=grouped_reference)}"
        )
    else:
        interval = "—"
    return row.get("display_flag", flag), row.get("display_interval", interval)


def _render_report(  # noqa: PLR0913
    *,
    patient_name: str,
    birth_date: date,
    sex: str,
    mrn: str,
    report_date: date,
    collection_date: date,
    accession: str,
    layout: str,
    rows: list[dict[str, Any]],
    short_names: dict[str, list[str]],
    comments_by_layout: dict[str, str],
    corrected: bool = False,
    cancelled: bool = False,
    cumulative_summary: bool = False,
) -> str:
    seed_accession = accession.replace("ACC-", "H2-", 1)
    letterhead = _LAB_LETTERHEADS[
        int.from_bytes(hashlib.sha256(seed_accession.encode()).digest()[:2], "big")
        % len(_LAB_LETTERHEADS)
    ]
    time_seed = int.from_bytes(
        hashlib.sha256(f"{seed_accession}:time".encode()).digest()[:2], "big"
    )
    collected_at = (
        f"{collection_date.isoformat()} {7 + time_seed % 3:02d}:{(time_seed // 7) % 60:02d}"
    )
    received_at = f"{report_date.isoformat()} {10 + time_seed % 3:02d}:{(time_seed // 11) % 60:02d}"
    reported_at = f"{report_date.isoformat()} {12 + time_seed % 4:02d}:{(time_seed // 13) % 60:02d}"
    lab_id = f"FIC-LAB-{100 + time_seed % 900:03d}"
    ordering_provider = _ORDERING_PROVIDERS[time_seed % len(_ORDERING_PROVIDERS)]
    lines = [
        letterhead[0],
        letterhead[1],
        f"Telephone: {letterhead[2]}   Laboratory ID: {lab_id}",
        "FINAL LABORATORY REPORT",
    ]
    if corrected:
        lines.append("*** CORRECTED REPORT — REPLACES THE PRIOR REPORT FOR THIS ACCESSION ***")
    if cumulative_summary:
        lines.append("CUMULATIVE SUMMARY")
    lines.extend(
        [
            "",
            "PATIENT / SPECIMEN",
            f"Patient: {patient_name}     DOB: {birth_date.isoformat()}     Sex: {sex.title()}",
            f"MRN: {mrn}     Ordering provider: {ordering_provider}",
            f"Accession: {accession}",
            f"Collected: {collected_at}",
            f"Received: {received_at}",
            f"Reported: {reported_at}",
            "",
            "RESULTS",
            f"{'Test':<36} {'Result':>9} {'Flag':>4}  {'Units':<20} {'Reference interval':>20}",
            "-" * 94,
        ]
    )
    layout_index = REPORT_LAYOUTS.index(layout)
    previous_section = None
    for row in rows:
        section = row.get("specimen_section")
        if section and section != previous_section:
            lines.append(section)
            previous_section = section
        short_name = short_names[row["code"]][layout_index % len(short_names[row["code"]])]
        result = row.get(
            "display_value",
            _format_value(row["value"], row["precision"]) if "value" in row else "",
        )
        flag, interval = _range_annotation(row)
        unit = row.get(
            "display_unit",
            _HUMAN_UNITS.get(row["unit"], row["unit"]) if row.get("unit") else "—",
        )
        lines.append(f"{short_name:<36} {result:>9} {flag:>4}  {unit:<20} {interval:>20}")
        if row["effective"] != collection_date.isoformat():
            lines.append(f"Add-on specimen collected: {row['effective']} — {short_name}")
    comment = comments_by_layout.get(layout)
    if comment:
        lines.extend(["", f"Comments: {comment.removeprefix('Comment: ')}"])
    if cancelled:
        lines.extend(["", "Comments: Potassium specimen hemolyzed; test not performed."])
    lines.extend(
        [
            "",
            f"Page 1 of 1 | {letterhead[0]}",
            "SYNTHETIC RECORD — NOT A REAL PATIENT",
        ]
    )
    return "\n".join(lines)


_LAB_LETTERHEADS = (
    (
        "Juniper Ridge Reference Laboratory",
        "4100 Example Way, Fairhaven, OR 97000",
        "(555) 010-1411",
    ),
    (
        "Blue Heron Clinical Reference Lab",
        "72 Fictional Park, Bellwether, MN 55000",
        "(555) 010-1834",
    ),
    (
        "Cedar Vale Diagnostic Laboratory",
        "900 Placeholder Road, Northfield, MA 01000",
        "(555) 010-1276",
    ),
)
_ORDERING_PROVIDERS = ("Jordan Lee, MD", "Casey Morgan, DO", "Riley Chen, MD", "Taylor Brooks, NP")
_HUMAN_UNITS = {
    "mL/min/{1.73_m2}": "mL/min/1.73 m²",
    "10*3/uL": "10³/µL",
}


def _format_value(value: float | int, precision: int, *, grouping: bool = False) -> str:
    if grouping:
        return f"{value:,.{precision}f}"
    return f"{value:.{precision}f}"


def _rounded_value(value: float, precision: int) -> float | int:
    rounded = round(value, precision)
    return int(rounded) if precision == 0 else rounded
