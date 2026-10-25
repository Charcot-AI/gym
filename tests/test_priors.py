from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from charcot_gym.synth.hard_clinical_coding import generate_hard_clinical_coding_task_sets
from charcot_gym.synth.hard_document_fhir import generate_hard_document_fhir_task_sets
from charcot_gym.synth.hard_multitool import prepare_hard_task
from charcot_gym.synth.priors import panel_weights

ROOT = Path(__file__).resolve().parents[1]


def _tiny_priors() -> dict[str, Any]:
    return {
        "condition_categories": [
            {"icd10_category": "I10", "patient_share": 1.0},
        ],
        "conditions_per_patient": {
            **{str(count): 0.0 for count in range(1, 10)},
            "1": 1.0,
            "10+": 0.0,
        },
        "oncology_patient_share": 0.0,
        "lab_frequency": [{"loinc": "2345-7", "share": 1.0}],
        "lab_value_percentiles": [
            {
                "loinc": "2345-7",
                "ucum": "mg/dL",
                "p05": 90.0,
                "p25": 100.0,
                "p50": 110.0,
                "p75": 120.0,
                "p95": 130.0,
            }
        ],
        "lab_pairs_same_encounter": [],
    }


@pytest.mark.parametrize("family", ["clinical_coding", "document_fhir"])
def test_hard_public_generation_without_priors_matches_committed_artifacts(
    family: str,
) -> None:
    if family == "clinical_coding":
        generated, _private = generate_hard_clinical_coding_task_sets(0, priors=None)
        task_directory = ROOT / "src" / "charcot_gym" / "tasks" / family
    else:
        generated, _private = generate_hard_document_fhir_task_sets(0, priors=None)
        generated = [prepare_hard_task(task) for task in generated]
        task_directory = ROOT / "src" / "charcot_gym" / "tasks" / family

    for task in generated:
        artifact = task_directory / f"{task['id']}.json"
        assert json.loads(artifact.read_text(encoding="utf-8")) == task


def test_synthetic_priors_shift_document_panels_and_routine_values_deterministically() -> None:
    priors = _tiny_priors()
    calibration = json.loads(
        (ROOT / "src" / "charcot_gym" / "synth" / "calibration.json").read_text(encoding="utf-8")
    )
    analytes = json.loads(
        (ROOT / "src" / "charcot_gym" / "data" / "loinc_subset.json").read_text(encoding="utf-8")
    )
    weights = panel_weights(calibration["panel_weights"], analytes, priors)
    assert weights["BMP"] > calibration["panel_weights"]["BMP"]

    first = generate_hard_document_fhir_task_sets(123, priors=priors)[1]
    second = generate_hard_document_fhir_task_sets(123, priors=priors)[1]
    assert first == second
    parameters = calibration["analyte_parameters"]["2345-7"]
    for task in first:
        if task["metadata"]["profile"] != "stable_routine":
            continue
        glucose = [item for item in task["gold"]["expected"] if item["code"] == "2345-7"]
        assert glucose
        for item in glucose:
            assert 90 <= item["value"] <= 130
            assert parameters["physiologic_min"] <= item["value"] <= parameters["physiologic_max"]
