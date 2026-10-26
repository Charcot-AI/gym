"""Clinical facts independent of their source presentation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

from charcot_gym.synth.generator import NAMES, _choose_panels
from charcot_gym.synth.lab_generation import _CurrentContext, _draw_current
from charcot_gym.synth.lab_values import _reference_range


@dataclass(frozen=True)
class LabResult:
    """One result with its patient-specific reference interval."""

    code: str
    display: str
    value: float | int
    unit: str
    low: float
    high: float
    flag: str
    comment: str = ""


@dataclass(frozen=True)
class LabReport:
    """Patient identity, specimen timing, and laboratory results."""

    name: str
    dob: str
    mrn: str
    sex: str
    accession: str
    collected: str
    reported: str
    results: tuple[LabResult, ...]
    scenario: str = "routine"


def preliminary_report(report: LabReport) -> LabReport:
    """Vary one chemistry result without disturbing coupled blood counts or lipids."""
    if not report.results:
        return report
    target = next(
        r for r in report.results if r.code in {"2345-7", "2160-0", "2823-3", "2951-2", "3094-0"}
    )
    calibration = json.loads((Path(__file__).resolve().parents[1] / "calibration.json").read_text())
    parameters = calibration["analyte_parameters"][target.code]
    for factor in (1.05, 0.95):
        value = round(target.value * factor, parameters["precision"])
        if (
            value != target.value
            and parameters["physiologic_min"] <= value <= parameters["physiologic_max"]
        ):
            return replace(
                report,
                results=tuple(
                    replace(r, value=value) if r.code == target.code else r for r in report.results
                ),
            )
    raise ValueError("Could not vary a preliminary chemistry result.")


def build_facts(seed: int) -> LabReport:
    """Draw patient-specific coupled panels from the bundled calibration."""
    rng = random.Random(seed)
    root = Path(__file__).resolve().parents[2]
    calibration = json.loads((root / "synth/calibration.json").read_text())
    analytes = json.loads((root / "data/loinc_subset.json").read_text())
    scenarios = json.loads((root / "synth/scenarios.json").read_text())["scenarios"]
    scenario = rng.choice(
        [s for s in scenarios if s["id"] in {"routine", "hypokalemia", "uncontrolled_diabetes"}]
    )
    panels = _choose_panels(rng, scenario["panels"], calibration) | {"CBC + ANC"}
    codes = {"718-7", "4544-3", "6690-2", "751-8"}
    codes.update(rng.sample(["2345-7", "2160-0"], rng.randint(1, 2)))
    analytes = [
        entry for entry in analytes if entry["code"] in codes or entry.get("panel") in panels
    ]
    for entry in analytes:
        entry["precision"] = calibration["analyte_parameters"][entry["code"]]["precision"]
    sex = rng.choice(["female", "male"])
    fraction = calibration["hematology_coupling"]["neutrophil_fraction"]
    for _ in range(10_000):
        current = _draw_current(
            _CurrentContext(rng, scenario, analytes, [], calibration, None, None, None, sex)
        )
        by_code = {item["entry"]["code"]: item["value"] for item in current}
        if fraction["min"] <= by_code["751-8"] / by_code["6690-2"] <= fraction["max"]:
            break
    else:
        raise ValueError("Could not sample a coupled routine blood count.")
    results = []
    for item in current:
        entry, value = item["entry"], item["value"]
        reference = _reference_range(entry, sex, calibration)
        low, high = reference["min"], reference["max"]
        results.append(
            LabResult(
                entry["code"],
                entry["display"],
                value,
                entry["unit"],
                low,
                high,
                "L" if value < low else "H" if value > high else "N",
                "Specimen quality acceptable.",
            )
        )
    rng.shuffle(results)
    month, day = rng.sample(range(1, 13), 2)
    collected = date(rng.randint(2023, 2025), month, day)
    reported = collected.replace(day=rng.randint(18, 28))
    return LabReport(
        " ".join(rng.choice(NAMES)),
        (date(1950, 1, 1) + timedelta(days=rng.randrange(20_000))).isoformat(),
        f"SYN-{seed}",
        sex,
        f"LAB-{seed}",
        collected.isoformat(),
        reported.isoformat(),
        tuple(results),
        scenario["id"],
    )
