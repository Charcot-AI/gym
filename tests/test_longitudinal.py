from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

from charcot_gym.synth.longitudinal import generate_longitudinal_record

ROOT = Path(__file__).resolve().parents[1]
SYNTH = ROOT / "src" / "charcot_gym" / "synth"
CALIBRATION = json.loads((SYNTH / "calibration.json").read_text(encoding="utf-8"))
ANALYTES = json.loads((ROOT / "src" / "charcot_gym" / "data" / "loinc_subset.json").read_text())
PROFILES = json.loads((SYNTH / "longitudinal.json").read_text(encoding="utf-8"))["profiles"]
LIPID_CODES = ("2093-3", "2085-9", "13457-7", "2571-8")
CBC_CODES = ("718-7", "4544-3", "6690-2", "751-8", "777-3")


def _record(profile_name: str, seed: int) -> list[dict[str, Any]]:
    sex = "female" if seed % 2 else "male"
    return generate_longitudinal_record(
        random.Random(seed),
        patient_id="patient-longitudinal",
        sex=sex,
        birth_date=date(1960, 4, 15),
        end_date=date(2025, 12, 31),
        years=3,
        profile=PROFILES[profile_name],
        calibration=CALIBRATION,
        analytes=ANALYTES,
    )


def _observation_groups(
    resources: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for resource in resources:
        if resource["resourceType"] != "Observation":
            continue
        code = resource["code"]["coding"][0]["code"]
        groups[resource["encounter"]["reference"]][code] = resource
    return groups


def _value(observation: dict[str, Any]) -> float:
    return float(observation["valueQuantity"]["value"])


def _age_at(birth_date: date, on_date: date) -> int:
    return (
        on_date.year
        - birth_date.year
        - ((on_date.month, on_date.day) < (birth_date.month, birth_date.day))
    )


def _ckd_epi_2021(creatinine: float, age: int, sex: str) -> float:
    kappa = 0.7 if sex == "female" else 0.9
    alpha = -0.241 if sex == "female" else -0.302
    ratio = creatinine / kappa
    return (
        142
        * min(ratio, 1) ** alpha
        * max(ratio, 1) ** -1.2
        * 0.9938**age
        * (1.012 if sex == "female" else 1)
    )


def test_longitudinal_profiles_satisfy_properties_over_200_seeds() -> None:
    for profile_name, profile in PROFILES.items():
        for seed in range(200):
            resources = _record(profile_name, seed)
            patient = next(item for item in resources if item["resourceType"] == "Patient")
            encounters = [item for item in resources if item["resourceType"] == "Encounter"]
            observations = [item for item in resources if item["resourceType"] == "Observation"]
            conditions = [item for item in resources if item["resourceType"] == "Condition"]
            encounter_ids = {item["id"] for item in encounters}
            assert 6 <= len(encounters) <= 14
            assert encounters
            assert all(item["class"][0]["coding"][0]["code"] == "AMB" for item in encounters)

            dated = []
            for resource in resources:
                if resource["resourceType"] == "Encounter":
                    dated.append(date.fromisoformat(resource["actualPeriod"]["start"]))
                elif resource["resourceType"] == "Observation":
                    dated.append(date.fromisoformat(resource["effectiveDateTime"]))
                elif resource["resourceType"] == "Condition":
                    dated.append(date.fromisoformat(resource["recordedDate"]))
            assert dated == sorted(dated)
            assert all(item <= date(2025, 12, 31) for item in dated)

            for observation in observations:
                assert observation["subject"]["reference"] == f"Patient/{patient['id']}"
                reference = observation["encounter"]["reference"]
                assert reference.startswith("Encounter/")
                assert reference.split("/", 1)[1] in encounter_ids
                code = observation["code"]["coding"][0]["code"]
                value = _value(observation)
                if code in CALIBRATION["analyte_parameters"]:
                    parameters = CALIBRATION["analyte_parameters"][code]
                    assert parameters["physiologic_min"] <= value <= parameters["physiologic_max"]
                    assert math.isclose(value, round(value, parameters["precision"]))
                elif profile_name == "ckd_progression":
                    egfr = profile["egfr"]
                    assert code == egfr["code"]
                    assert egfr["physiologic_min"] <= value <= egfr["physiologic_max"]
                    assert math.isclose(value, round(value, egfr["precision"]))

            for condition in conditions:
                assert condition["subject"]["reference"] == f"Patient/{patient['id']}"
                assert condition["clinicalStatus"]["coding"][0]["code"] in {"active", "resolved"}
                assert condition["encounter"]["reference"].split("/", 1)[1] in encounter_ids

            groups = _observation_groups(resources)
            assert len(groups) == len(encounters)
            for group in groups.values():
                assert set(LIPID_CODES) <= set(group)
                total = _value(group["2093-3"])
                hdl = _value(group["2085-9"])
                ldl = _value(group["13457-7"])
                triglycerides = _value(group["2571-8"])
                assert abs(total - (ldl + hdl + triglycerides / 5)) <= 5
                assert set(CBC_CODES) <= set(group)
                assert 2.88 <= _value(group["4544-3"]) / _value(group["718-7"]) <= 3.12

                if profile_name == "ckd_progression":
                    creatinine = _value(group["2160-0"])
                    egfr = _value(group[profile["egfr"]["code"]])
                    observed_date = date.fromisoformat(group["2160-0"]["effectiveDateTime"])
                    age = _age_at(date(1960, 4, 15), observed_date)
                    sex = patient["gender"]
                    expected = _ckd_epi_2021(creatinine, age, sex)
                    assert abs(egfr - expected) / expected <= 0.10

                if profile_name == "diabetes_course":
                    hba1c = _value(group[profile["hba1c"]["code"]])
                    glucose = _value(group[profile["hba1c"]["glucose_code"]])
                    estimated_average_glucose = 28.7 * hba1c - 46.7
                    assert abs(glucose - estimated_average_glucose) <= 25

            if profile_name == "oncology_course":
                ordered_groups = [groups[f"Encounter/{item['id']}"] for item in encounters]
                for index in range(0, len(ordered_groups) - 1, 2):
                    nadir_wbc = _value(ordered_groups[index]["6690-2"])
                    recovery_wbc = _value(ordered_groups[index + 1]["6690-2"])
                    assert nadir_wbc < recovery_wbc
                    anc = _value(ordered_groups[index]["751-8"])
                    assert 0.35 <= anc / nadir_wbc <= 0.8
            elif profile_name == "ckd_progression":
                egfr_values = [_value(group[profile["egfr"]["code"]]) for group in groups.values()]
                assert egfr_values[0] >= 60
                assert any(45 <= value < 60 for value in egfr_values)
                assert any(30 <= value < 45 for value in egfr_values)
                assert any(value < 30 for value in egfr_values)


def test_longitudinal_output_is_deterministic() -> None:
    assert _record("ckd_progression", 128) == _record("ckd_progression", 128)


def test_ckd_profile_crosses_stage_boundaries() -> None:
    resources = _record("ckd_progression", 23)
    observations = _observation_groups(resources)
    egfr_values = [
        _value(group[PROFILES["ckd_progression"]["egfr"]["code"]])
        for group in observations.values()
    ]
    assert egfr_values[0] >= 60
    assert any(45 <= value < 60 for value in egfr_values)
    assert any(30 <= value < 45 for value in egfr_values)
    assert any(value < 30 for value in egfr_values)
