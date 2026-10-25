"""Generation of deterministic synthetic longitudinal clinical records."""

from __future__ import annotations

import random
from datetime import date, timedelta
from typing import Any

from charcot_gym.synth.lab_values import (
    _draw_lipid_panel,
    _draw_value,
    _enforce_cbc_coupling,
    _round_value,
    _sample_value,
)

LOINC_SYSTEM = "http://loinc.org"
UCUM_SYSTEM = "http://unitsofmeasure.org"
CONDITION_CLINICAL_SYSTEM = "http://terminology.hl7.org/CodeSystem/condition-clinical"
ACT_CODE_SYSTEM = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
ICD10_SYSTEM = "http://hl7.org/fhir/sid/icd-10-cm"


def generate_longitudinal_record(  # noqa: PLR0913
    rng: random.Random,
    *,
    patient_id: str,
    sex: str,
    birth_date: date,
    end_date: date,
    years: int,
    profile: dict[str, Any],
    calibration: dict[str, Any],
    analytes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a linked patient record using the configured longitudinal profile."""
    if sex not in {"female", "male"}:
        raise ValueError("sex must be female or male")
    if not isinstance(years, int) or isinstance(years, bool) or years < 1:
        raise ValueError("years must be a positive integer")
    if end_date < birth_date:
        raise ValueError("end_date must not precede birth_date")

    profile_kind = profile.get("kind")
    if profile_kind not in {
        "ckd_progression",
        "diabetes_course",
        "oncology_course",
        "stable_routine",
    }:
        raise ValueError(f"unsupported longitudinal profile: {profile_kind!r}")

    count_config = profile["encounter_count"]
    minimum = int(count_config["min"])
    maximum = int(count_config["max"])
    if not 6 <= minimum <= maximum <= 14:
        raise ValueError("encounter_count must be between 6 and 14")
    encounter_count = rng.randint(minimum, maximum)
    dates, phases = _encounter_dates(rng, end_date, years, encounter_count, profile)

    by_code: dict[str, dict[str, Any]] = {}
    for source_entry in analytes:
        entry = dict(source_entry)
        parameters = calibration["analyte_parameters"].get(entry["code"])
        if parameters is not None:
            entry["precision"] = parameters["precision"]
        by_code[entry["code"]] = entry

    required_codes = set(profile["lipid_codes"]) | set(profile["cbc_codes"])
    required_codes.update(profile.get("additional_codes", []))
    if profile_kind == "ckd_progression":
        required_codes.add("2160-0")
    if profile_kind == "diabetes_course":
        required_codes.update({profile["hba1c"]["code"], profile["hba1c"]["glucose_code"]})
    missing_codes = required_codes - set(by_code)
    if missing_codes:
        raise ValueError(
            f"missing analytes required by profile: {', '.join(sorted(missing_codes))}"
        )

    patient = {
        "resourceType": "Patient",
        "id": patient_id,
        "gender": sex,
        "birthDate": birth_date.isoformat(),
    }
    results: list[dict[str, Any]] = [patient]
    conditions_by_index: dict[int, list[dict[str, Any]]] = {}
    for condition in profile.get("conditions", []):
        index = round(float(condition["at_fraction"]) * (encounter_count - 1))
        conditions_by_index.setdefault(index, []).append(condition)

    observation_index = 0
    for index, effective_date in enumerate(dates):
        encounter_id = f"encounter-l-{index + 1:03d}"
        encounter = {
            "resourceType": "Encounter",
            "id": encounter_id,
            "status": "completed",
            "class": [{"coding": [{"system": ACT_CODE_SYSTEM, "code": "AMB"}]}],
            "subject": {"reference": f"Patient/{patient_id}"},
            "actualPeriod": {"start": effective_date.isoformat()},
        }
        encounter_reference = {"reference": f"Encounter/{encounter_id}"}
        values = _draw_laboratory_values(
            rng,
            index,
            encounter_count,
            phases[index],
            effective_date,
            birth_date,
            sex,
            profile,
            profile_kind,
            calibration,
            by_code,
        )
        observations = []
        for code, value in sorted(values.items()):
            entry_for_observation: dict[str, Any] | None = by_code.get(code)
            if entry_for_observation is None and profile_kind == "ckd_progression":
                entry_for_observation = _egfr_entry(profile["egfr"])
            assert entry_for_observation is not None
            observation_index += 1
            observations.append(
                _observation(
                    patient_id,
                    encounter_reference,
                    entry_for_observation,
                    value,
                    effective_date,
                    observation_index,
                )
            )
        results.append(encounter)
        results.extend(observations)
        for condition in conditions_by_index.get(index, []):
            results.append(
                _condition(
                    patient_id,
                    encounter_reference,
                    condition,
                    effective_date,
                    index,
                )
            )
    return results


def _encounter_dates(
    rng: random.Random,
    end_date: date,
    years: int,
    count: int,
    profile: dict[str, Any],
) -> tuple[list[date], list[str | None]]:
    if profile["kind"] == "oncology_course":
        cycle_length = int(profile["cycles"]["length_days"])
        nadir_day = int(profile["cycles"]["nadir_day"])
        cycles = (count + 1) // 2
        start = end_date - timedelta(days=cycle_length * cycles)
        dates: list[date] = []
        phases: list[str | None] = []
        for index in range(count):
            cycle = index // 2
            is_nadir = index % 2 == 0
            day = nadir_day if is_nadir else cycle_length
            dates.append(start + timedelta(days=cycle * cycle_length + day))
            phases.append("nadir" if is_nadir else "recovery")
        return dates, phases

    span = years * 365
    offsets = sorted(rng.sample(range(span + 1), count))
    start = end_date - timedelta(days=span)
    return [start + timedelta(days=offset) for offset in offsets], [None] * count


def _draw_laboratory_values(  # noqa: PLR0913, PLR0917
    rng: random.Random,
    index: int,
    encounter_count: int,
    phase: str | None,
    effective_date: date,
    birth_date: date,
    sex: str,
    profile: dict[str, Any],
    profile_kind: str,
    calibration: dict[str, Any],
    by_code: dict[str, dict[str, Any]],
) -> dict[str, float | int]:
    values: dict[str, float | int] = {}
    normal_scenario: dict[str, Any] = {"abnormal_analytes": []}
    lipid_entries = [by_code[code] for code in profile["lipid_codes"]]
    values.update(_draw_lipid_panel(rng, lipid_entries, {}, calibration, sex, normal_scenario))

    cbc_entries = [by_code[code] for code in profile["cbc_codes"]]
    cbc_values: list[dict[str, Any]] = []
    for entry in cbc_entries:
        code = entry["code"]
        cbc_range = profile.get("cbc_ranges", {}).get(phase, {}).get(code)
        if cbc_range is None:
            result = _draw_value(rng, entry, None, calibration, sex, False)
        else:
            result = _draw_value(rng, entry, cbc_range, calibration, sex, True)
        cbc_values.append({"entry": entry, "value": result})
    coupling_scenario = {
        "abnormal_analytes": profile.get("abnormal_analytes", [])
        if profile_kind == "oncology_course"
        else []
    }
    _enforce_cbc_coupling(cbc_values, rng, calibration, coupling_scenario, sex)
    values.update({item["entry"]["code"]: item["value"] for item in cbc_values})

    for code in profile.get("additional_codes", []):
        if code in values:
            continue
        entry = by_code[code]
        value_range = profile.get("bun_range") if code == "3094-0" else None
        values[code] = _draw_value(
            rng,
            entry,
            value_range,
            calibration,
            sex,
            value_range is not None,
        )

    if profile_kind == "ckd_progression":
        target = _progression_target(rng, index, encounter_count, profile["egfr"])
        age = _age_at(birth_date, effective_date)
        creatinine = _creatinine_for_egfr(target, age, sex)
        creatinine_entry = by_code["2160-0"]
        creatinine = _round_value(creatinine, creatinine_entry["precision"])
        _check_physiologic_value(creatinine, creatinine_entry, calibration)
        values["2160-0"] = creatinine
        egfr = _round_value(_ckd_epi_2021(creatinine, age, sex), profile["egfr"]["precision"])
        if not profile["egfr"]["physiologic_min"] <= egfr <= profile["egfr"]["physiologic_max"]:
            raise ValueError("generated eGFR is outside configured physiological bounds")
        values[profile["egfr"]["code"]] = egfr
    elif profile_kind == "diabetes_course":
        config = profile["hba1c"]
        entry = by_code[config["code"]]
        nominal = config["start"] + (config["end"] - config["start"]) * index / (
            encounter_count - 1
        )
        value = _sample_value(
            rng,
            nominal - config["jitter"],
            nominal + config["jitter"],
            entry["precision"],
        )
        _check_physiologic_value(value, entry, calibration)
        values[config["code"]] = value
        mean_glucose = 28.7 * value - 46.7
        glucose_entry = by_code[config["glucose_code"]]
        glucose = _round_value(
            mean_glucose + rng.uniform(-config["glucose_jitter"], config["glucose_jitter"]),
            glucose_entry["precision"],
        )
        _check_physiologic_value(glucose, glucose_entry, calibration)
        values[config["glucose_code"]] = glucose

    if profile_kind == "stable_routine":
        for code in profile.get("additional_codes", []):
            entry = by_code[code]
            _check_physiologic_value(values[code], entry, calibration)
    return values


def _progression_target(
    rng: random.Random, index: int, count: int, config: dict[str, Any]
) -> float:
    if index == 0:
        return float(config["start"])
    if index == count - 1:
        return float(config["end"])
    target = float(config["start"]) + (float(config["end"]) - float(config["start"])) * index / (
        count - 1
    )
    return target + rng.uniform(-float(config["jitter"]), float(config["jitter"]))


def _creatinine_for_egfr(egfr: float, age: int, sex: str) -> float:
    kappa = 0.7 if sex == "female" else 0.9
    sex_factor = 1.012 if sex == "female" else 1.0
    baseline = 142 * (0.9938**age) * sex_factor
    return float(kappa * (egfr / baseline) ** (-1 / 1.2))


def _ckd_epi_2021(creatinine: float, age: int, sex: str) -> float:
    kappa = 0.7 if sex == "female" else 0.9
    alpha = -0.241 if sex == "female" else -0.302
    ratio = creatinine / kappa
    sex_factor = 1.012 if sex == "female" else 1.0
    return float(142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.2 * 0.9938**age * sex_factor)


def _age_at(birth_date: date, on_date: date) -> int:
    return (
        on_date.year
        - birth_date.year
        - ((on_date.month, on_date.day) < (birth_date.month, birth_date.day))
    )


def _check_physiologic_value(
    value: float | int, entry: dict[str, Any], calibration: dict[str, Any]
) -> None:
    parameters = calibration["analyte_parameters"][entry["code"]]
    if (
        value < parameters["physiologic_min"]
        or value > parameters["physiologic_max"]
        or value != _round_value(value, parameters["precision"])
    ):
        raise ValueError(f"generated value for {entry['code']} violates its calibration")


def _egfr_entry(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "system": LOINC_SYSTEM,
        "code": config["code"],
        "display": config["display"],
        "unit": config["unit"],
        "precision": config["precision"],
    }


def _observation(  # noqa: PLR0913, PLR0917
    patient_id: str,
    encounter_reference: dict[str, str],
    entry: dict[str, Any],
    value: float | int,
    effective_date: date,
    observation_index: int,
) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": f"observation-l-{observation_index:03d}",
        "status": "final",
        "code": {
            "coding": [
                {
                    "system": entry.get("system", LOINC_SYSTEM),
                    "code": entry["code"],
                    "display": entry.get("display", entry["code"]),
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": dict(encounter_reference),
        "effectiveDateTime": effective_date.isoformat(),
        "valueQuantity": {
            "value": value,
            "unit": entry["unit"],
            "system": UCUM_SYSTEM,
            "code": entry["unit"],
        },
    }


def _condition(
    patient_id: str,
    encounter_reference: dict[str, str],
    condition: dict[str, Any],
    recorded_date: date,
    encounter_index: int,
) -> dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": f"condition-l-{encounter_index + 1:03d}-{condition['code']}",
        "clinicalStatus": {
            "coding": [
                {
                    "system": CONDITION_CLINICAL_SYSTEM,
                    "code": condition["clinical_status"],
                }
            ]
        },
        "code": {
            "coding": [
                {
                    "system": ICD10_SYSTEM,
                    "code": condition["code"],
                    "display": condition["display"],
                }
            ]
        },
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": dict(encounter_reference),
        "recordedDate": recorded_date.isoformat(),
    }
