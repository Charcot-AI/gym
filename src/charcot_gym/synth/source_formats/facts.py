"""Clinical facts independent of their source presentation."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

from charcot_gym.synth.generator import NAMES
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


def build_facts(seed: int) -> LabReport:
    """Draw a coupled routine panel from the bundled calibration."""
    rng = random.Random(seed)
    root = Path(__file__).resolve().parents[2]
    calibration = json.loads((root / "synth/calibration.json").read_text())
    analytes = json.loads((root / "data/loinc_subset.json").read_text())
    codes = {"2345-7", "2160-0", "2823-3", "718-7", "4544-3", "6690-2", "751-8"}
    analytes = [entry for entry in analytes if entry["code"] in codes]
    for entry in analytes:
        entry["precision"] = calibration["analyte_parameters"][entry["code"]]["precision"]
    sex = rng.choice(["female", "male"])
    scenario = {"id": "routine", "current_rules": {}, "abnormal_analytes": []}
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
    collected = date(2025, 1, 1) + timedelta(days=rng.randrange(300))
    return LabReport(
        " ".join(rng.choice(NAMES)),
        "1982-03-14",
        f"SYN-{seed}",
        sex,
        f"LAB-{seed}",
        collected.isoformat(),
        (collected + timedelta(days=1)).isoformat(),
        tuple(results),
    )
