from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any

import pytest

from charcot_gym import (
    ClinicalEnv,
    Submit,
    Task,
    ToolCall,
    load_tasks,
    oracle_answer,
    oracle_turns,
)
from charcot_gym.scores import SCORES, ScoreInputError, calculate, score_spec
from charcot_gym.synth.clinical_scores import generate_clinical_score_task_sets
from charcot_gym.verifier import Score

# Expected values are worked by hand from each publication's equation or point table.
FORMULA_CASES: list[tuple[str, dict[str, Any], float, str | None, str]] = [
    ("bmi", {"weight_kg": 70, "height_cm": 175}, 22.9, "normal", "Keys 1972; WHO TRS 894"),
    ("bmi", {"weight_kg": 95, "height_cm": 170}, 32.9, "obese", "Keys 1972; WHO TRS 894"),
    ("bsa_mosteller", {"height_cm": 180, "weight_kg": 80}, 2.0, None, "Mosteller 1987"),
    ("qtc_bazett", {"qt_ms": 400, "heart_rate": 60}, 400, None, "Bazett 1920"),
    ("qtc_bazett", {"qt_ms": 400, "heart_rate": 75}, 447, None, "Bazett 1920"),
    ("qtc_fridericia", {"qt_ms": 400, "heart_rate": 75}, 431, None, "Fridericia 1920"),
    ("anion_gap", {"sodium": 140, "chloride": 104, "bicarbonate": 24}, 12.0, None, "Emmett 1977"),
    ("corrected_calcium", {"calcium_mg_dl": 8.0, "albumin_g_dl": 2.0}, 9.6, None, "Payne 1973"),
    ("corrected_sodium_katz", {"sodium": 130, "glucose_mg_dl": 600}, 138.0, None, "Katz 1973"),
    (
        "egfr_ckd_epi_2021",
        {"age": 50, "sex": "female", "creatinine_mg_dl": 1.0},
        69,
        "G2",
        "Inker 2021",
    ),
    (
        "egfr_ckd_epi_2021",
        {"age": 60, "sex": "male", "creatinine_mg_dl": 0.9},
        98,
        "G1",
        "Inker 2021",
    ),
    (
        "crcl_cockcroft_gault",
        {"age": 60, "sex": "male", "weight_kg": 72, "creatinine_mg_dl": 1.0},
        80,
        None,
        "Cockcroft 1976",
    ),
    (
        "crcl_cockcroft_gault",
        {"age": 60, "sex": "female", "weight_kg": 72, "creatinine_mg_dl": 1.0},
        68,
        None,
        "Cockcroft 1976",
    ),
    (
        "cha2ds2_vasc",
        {
            "age": 76,
            "sex": "female",
            "congestive_heart_failure": False,
            "hypertension": True,
            "diabetes": True,
            "prior_stroke_tia_thromboembolism": False,
            "vascular_disease": False,
        },
        5,
        "high",
        "Lip 2010",
    ),
    (
        "has_bled",
        {
            "hypertension": True,
            "abnormal_renal_function": False,
            "abnormal_liver_function": False,
            "prior_stroke": False,
            "bleeding_history": True,
            "labile_inr": False,
            "elderly": True,
            "antiplatelet_or_nsaid": False,
            "alcohol_excess": False,
        },
        3,
        "high",
        "Pisters 2010",
    ),
    (
        "wells_pe",
        {
            "clinical_signs_dvt": True,
            "pe_most_likely_diagnosis": False,
            "heart_rate_over_100": True,
            "immobilization_or_surgery_4_weeks": False,
            "previous_dvt_or_pe": False,
            "hemoptysis": False,
            "malignancy": False,
        },
        4.5,
        "moderate",
        "Wells 2000",
    ),
    (
        "wells_dvt",
        {
            "active_cancer": True,
            "paralysis_or_recent_cast": False,
            "bedridden_or_major_surgery": False,
            "deep_vein_tenderness": False,
            "entire_leg_swollen": True,
            "calf_swelling_over_3_cm": False,
            "pitting_edema_symptomatic_leg": False,
            "collateral_superficial_veins": False,
            "previous_dvt": False,
            "alternative_diagnosis_as_likely": True,
        },
        0,
        "unlikely",
        "Wells 2003",
    ),
    (
        "perc",
        {
            "age": 40,
            "heart_rate": 80,
            "oxygen_saturation": 98,
            "unilateral_leg_swelling": False,
            "hemoptysis": False,
            "recent_surgery_or_trauma": False,
            "prior_dvt_or_pe": False,
            "hormone_use": False,
        },
        0,
        "negative",
        "Kline 2004",
    ),
    (
        "revised_geneva",
        {
            "age": 70,
            "heart_rate": 100,
            "previous_dvt_or_pe": False,
            "surgery_or_fracture_1_month": False,
            "active_malignancy": False,
            "unilateral_lower_limb_pain": False,
            "hemoptysis": True,
            "deep_vein_palpation_pain_and_edema": False,
        },
        8,
        "intermediate",
        "Le Gal 2006",
    ),
    (
        "heart",
        {"history": 2, "ecg": 1, "age": 50, "risk_factors": 1, "troponin": 0},
        5,
        "moderate",
        "Six 2008",
    ),
    (
        "timi_ua_nstemi",
        {
            "age": 70,
            "three_or_more_cad_risk_factors": False,
            "known_cad_stenosis_50": False,
            "aspirin_last_7_days": True,
            "severe_angina_2_episodes_24h": False,
            "st_deviation_0_5_mm": True,
            "positive_cardiac_marker": False,
        },
        3,
        None,
        "Antman 2000",
    ),
    (
        "curb_65",
        {
            "confusion": True,
            "urea_mmol_l": 8,
            "respiratory_rate": 20,
            "systolic_bp": 120,
            "diastolic_bp": 70,
            "age": 70,
        },
        3,
        "high",
        "Lim 2003",
    ),
    (
        "qsofa",
        {"respiratory_rate": 24, "altered_mentation": False, "systolic_bp": 95},
        2,
        "positive",
        "Seymour 2016",
    ),
    (
        "child_pugh",
        {
            "bilirubin_mg_dl": 2.5,
            "albumin_g_dl": 3.0,
            "prothrombin_prolongation_s": 3,
            "ascites": "slight",
            "encephalopathy": "none",
        },
        8,
        "B",
        "Pugh 1973",
    ),
    (
        "meld_na",
        {
            "bilirubin_mg_dl": 1.0,
            "inr": 1.0,
            "creatinine_mg_dl": 1.0,
            "sodium": 140,
            "dialysis_twice_past_week": False,
        },
        6.4,
        None,
        "Kim 2008; Kamath 2001",
    ),
    (
        "meld_na",
        {
            "bilirubin_mg_dl": 3.0,
            "inr": 2.0,
            "creatinine_mg_dl": 2.0,
            "sodium": 130,
            "dialysis_twice_past_week": False,
        },
        28.7,
        None,
        "Kim 2008; Kamath 2001",
    ),
    (
        "fib_4",
        {"age": 50, "ast": 40, "alt": 25, "platelets": 200},
        2.0,
        "indeterminate",
        "Sterling 2006",
    ),
    ("apri", {"ast": 80, "ast_upper_limit": 40, "platelets": 100}, 2.0, "high", "Wai 2003"),
    (
        "abcd2",
        {
            "age": 65,
            "systolic_bp": 150,
            "diastolic_bp": 80,
            "clinical_features": "unilateral_weakness",
            "duration_min": 70,
            "diabetes": True,
        },
        7,
        "high",
        "Johnston 2007",
    ),
    (
        "centor",
        {
            "tonsillar_exudate": True,
            "tender_anterior_cervical_nodes": True,
            "fever_history": True,
            "cough_absent": True,
        },
        4,
        None,
        "Centor 1981",
    ),
    (
        "mcisaac",
        {
            "age": 50,
            "tonsillar_exudate": True,
            "tender_anterior_cervical_nodes": True,
            "fever_history": True,
            "cough_absent": True,
        },
        3,
        None,
        "McIsaac 1998",
    ),
    ("gcs", {"eye": 3, "verbal": 4, "motor": 5}, 12, None, "Teasdale 1974"),
    ("phq_9", {f"item_{i}": 1 for i in range(1, 10)}, 9, "mild", "Kroenke 2001"),
    ("gad_7", {f"item_{i}": 2 for i in range(1, 8)}, 14, "moderate", "Spitzer 2006"),
    ("audit_c", {"item_1": 2, "item_2": 1, "item_3": 1}, 4, None, "Bush 1998"),
]

BAND_BOUNDARIES: list[tuple[str, dict[str, Any], str]] = [
    ("bmi", {"weight_kg": 73.9, "height_cm": 200}, "underweight"),
    ("bmi", {"weight_kg": 74.0, "height_cm": 200}, "normal"),
    ("bmi", {"weight_kg": 100.0, "height_cm": 200}, "overweight"),
    ("bmi", {"weight_kg": 120.0, "height_cm": 200}, "obese"),
    ("phq_9", {f"item_{i}": 1 if i <= 4 else 0 for i in range(1, 10)}, "minimal"),
    ("phq_9", {f"item_{i}": 1 if i <= 5 else 0 for i in range(1, 10)}, "mild"),
    ("phq_9", {f"item_{i}": 3 if i <= 5 else 0 for i in range(1, 10)}, "moderately_severe"),
    ("phq_9", {f"item_{i}": 3 if i <= 6 else 2 for i in range(1, 10)}, "severe"),
    ("gad_7", {f"item_{i}": 3 if i <= 5 else 0 for i in range(1, 8)}, "severe"),
    ("fib_4", {"age": 28, "ast": 20, "alt": 25, "platelets": 80}, "low"),
    ("fib_4", {"age": 29, "ast": 20, "alt": 25, "platelets": 80}, "indeterminate"),
    ("fib_4", {"age": 50, "ast": 66, "alt": 25, "platelets": 200}, "high"),
    ("apri", {"ast": 20, "ast_upper_limit": 40, "platelets": 100}, "low"),
    ("qsofa", {"respiratory_rate": 22, "altered_mentation": False, "systolic_bp": 101}, "negative"),
    ("qsofa", {"respiratory_rate": 22, "altered_mentation": False, "systolic_bp": 100}, "positive"),
    ("egfr_ckd_epi_2021", {"age": 80, "sex": "male", "creatinine_mg_dl": 6.0}, "G5"),
]


def score_tasks() -> list[Task]:
    return [task for task in load_tasks() if task.family == "clinical_scores"]


def public_task(scenario: str) -> Task:
    return next(
        task
        for task in score_tasks()
        if task.split == "public" and task.metadata["scenario_id"] == scenario
    )


def run(task: Task, calls: list[ToolCall], answer: dict[str, Any] | None) -> Score:
    environment = ClinicalEnv([task])
    environment.reset(task.id, seed=task.seed)
    for call in calls:
        environment.step(call)
    environment.step(Submit(answer=answer))
    return environment.score()


def oracle_calls(task: Task) -> list[ToolCall]:
    return [deepcopy(call) for turn in oracle_turns(task) for call in turn]


def score_observation(calls: list[ToolCall], score_id: str) -> dict[str, Any]:
    code = SCORES[score_id].code
    resource: dict[str, Any] = next(
        call.arguments["resource"]
        for call in calls
        if call.name == "fhir_create"
        and call.arguments["resource"]["resourceType"] == "Observation"
        and call.arguments["resource"]["code"]["coding"][0]["code"] == code
    )
    return resource


def check(score: Score, name: str) -> bool:
    return next(item.passed for item in score.checks if item.name == name)


def chart_resource(task: Task, reference: str) -> dict[str, Any]:
    resource_type, resource_id = reference.split("/")
    resource: dict[str, Any] = next(
        entry["resource"]
        for entry in task.initial_state["entry"]
        if entry["resource"]["resourceType"] == resource_type
        and entry["resource"]["id"] == resource_id
    )
    return resource


@pytest.mark.parametrize(
    ("score_id", "inputs", "value", "band", "source"),
    FORMULA_CASES,
    ids=[f"{case[0]}-{index}" for index, case in enumerate(FORMULA_CASES)],
)
def test_formula_matches_publication(
    score_id: str, inputs: dict[str, Any], value: float, band: str | None, source: str
) -> None:
    result = calculate(score_id, inputs)
    assert result["value"] == pytest.approx(value, abs=SCORES[score_id].tolerance or 1e-9), source
    assert result["band"] == band, source


def test_every_score_has_a_formula_case_and_citation() -> None:
    assert {case[0] for case in FORMULA_CASES} == set(SCORES)
    for score in SCORES.values():
        assert score.citation and score.description
        assert re.search(r"\b(19|20)\d{2}\b", score.citation), score.id


@pytest.mark.parametrize(("score_id", "inputs", "band"), BAND_BOUNDARIES)
def test_band_boundaries(score_id: str, inputs: dict[str, Any], band: str) -> None:
    assert calculate(score_id, inputs)["band"] == band


def test_calculator_never_defaults_missing_or_invalid_fields() -> None:
    with pytest.raises(ScoreInputError) as missing:
        calculate("centor", {"tonsillar_exudate": True, "fever_history": False})
    assert missing.value.missing == ["tender_anterior_cervical_nodes", "cough_absent"]
    with pytest.raises(ScoreInputError) as invalid:
        calculate("bmi", {"weight_kg": "70", "height_cm": 0, "weight_lb": 150})
    assert invalid.value.invalid == ["weight_lb", "weight_kg", "height_cm"]
    with pytest.raises(ScoreInputError) as unknown:
        calculate("not_a_score", {})
    assert unknown.value.code == "unknown_score"
    with pytest.raises(ScoreInputError):
        calculate("phq_9", {f"item_{i}": 4 for i in range(1, 10)})


def test_tool_returns_spec_and_structured_errors() -> None:
    task = public_task("anthropometrics")
    environment = ClinicalEnv([task])
    environment.reset(task.id)

    def call(arguments: dict[str, Any]) -> Any:
        return environment.step(
            ToolCall(name="clinical_score", arguments=arguments)
        ).observation.result

    spec = call({"operation": "spec", "score": "bmi"})
    assert spec == score_spec("bmi")
    assert [field["name"] for field in spec["fields"]] == ["weight_kg", "height_cm"]
    assert spec["fields"][0]["unit"] == "kg"
    assert call({"operation": "calculate", "score": "bmi", "inputs": {"weight_kg": 70}}) == {
        "error": {
            "code": "invalid_inputs",
            "message": "bmi needs valid values for every field",
            "missing": ["height_cm"],
            "invalid": [],
        }
    }
    assert call({"operation": "spec", "score": "nope"})["error"]["code"] == "unknown_score"
    assert call(
        {"operation": "calculate", "score": "bmi", "inputs": {"weight_kg": 70, "height_cm": 175}}
    ) == {"score": "bmi", "value": 22.9, "band": "normal", "units": "kg/m2"}


def test_generation_is_deterministic() -> None:
    first = generate_clinical_score_task_sets(11)
    second = generate_clinical_score_task_sets(11)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(first[1]) != json.dumps(generate_clinical_score_task_sets(12)[1])


def test_committed_sample_is_small_and_covers_every_trap() -> None:
    public = [task for task in score_tasks() if task.split == "public"]
    assert 0 < len(public) <= 10
    traps = {trap for task in public for trap in task.metadata["traps"]}
    assert {
        "missing_input",
        "unit_mismatch",
        "stale_value",
        "similar_score",
        "stated_wrong_score",
        "questionnaire",
    } <= traps


@pytest.mark.parametrize(
    "task",
    [task for task in score_tasks() if task.split == "public"],
    ids=lambda task: task.id,
)
def test_correct_state_without_calculator_scores_one(task: Task) -> None:
    calls = [call for call in oracle_calls(task) if call.name != "clinical_score"]
    assert run(task, calls, oracle_answer(task)).value == 1.0


def test_copying_a_wrong_score_from_a_note_scores_zero() -> None:
    task = public_task("af_risk")
    calls = oracle_calls(task)
    stated = task.gold["note_values"]["cha2ds2_vasc"]
    assert stated != next(
        item["value"] for item in task.gold["scores"] if item["score"] == "cha2ds2_vasc"
    )
    score_observation(calls, "cha2ds2_vasc")["valueInteger"] = stated
    score = run(task, calls, oracle_answer(task))
    assert not check(score, "cha2ds2_vasc")
    assert check(score, "has_bled")


def test_similar_score_does_not_replace_requested_score() -> None:
    task = public_task("pe_workup")
    calls = oracle_calls(task)
    observation = score_observation(calls, "revised_geneva")
    observation["code"]["coding"] = [
        {"system": SCORES["wells_pe"].code_system, "code": SCORES["wells_pe"].code}
    ]
    observation["valueQuantity"] = {
        "value": task.gold["note_values"]["wells_pe"],
        "unit": "{score}",
    }
    observation.pop("valueInteger")
    assert run(task, calls, oracle_answer(task)).value == 0.0


def test_guessing_a_missing_input_scores_zero() -> None:
    task = public_task("liver_panel")
    assert task.gold["missing_inputs"] == {"meld_na": ["inr"]}
    calls = oracle_calls(task)
    guessed = deepcopy(score_observation(calls, "fib_4"))
    guessed["code"]["coding"] = [
        {"system": SCORES["meld_na"].code_system, "code": SCORES["meld_na"].code}
    ]
    guessed["valueQuantity"]["value"] = 12.0
    guessed.pop("interpretation")
    calls.append(ToolCall(name="fhir_create", arguments={"resource": guessed}))
    score = run(task, calls, oracle_answer(task))
    assert not check(score, "meld_na")
    assert check(score, "fib_4")


@pytest.mark.parametrize(
    "answer",
    [None, {"missing_inputs": {"meld_na": []}}, {"missing_inputs": {"meld_na": ["inr", "sodium"]}}],
)
def test_missing_inputs_must_be_reported_exactly(answer: dict[str, Any] | None) -> None:
    task = public_task("liver_panel")
    assert not check(run(task, oracle_calls(task), answer), "meld_na")


def test_reporting_documented_score_as_missing_fails() -> None:
    task = public_task("anthropometrics")
    score = run(task, oracle_calls(task), {"missing_inputs": {"bmi": ["weight_kg"]}})
    assert not check(score, "bmi")


def test_unconverted_units_score_zero() -> None:
    task = public_task("anthropometrics")
    item = next(item for item in task.gold["scores"] if item["score"] == "bmi")
    weight = next(
        resource
        for reference in item["derived_from"]
        if (resource := chart_resource(task, reference))["code"]["coding"][0]["code"] == "29463-7"
    )
    assert weight["valueQuantity"]["unit"] == "[lb_av]"
    calls = oracle_calls(task)
    raw = calculate("bmi", {**item["inputs"], "weight_kg": weight["valueQuantity"]["value"]})
    score_observation(calls, "bmi")["valueQuantity"]["value"] = raw["value"]
    assert not check(run(task, calls, oracle_answer(task)), "bmi")


def test_stale_inputs_score_zero() -> None:
    task = public_task("renal_dosing")
    item = next(item for item in task.gold["scores"] if item["score"] == "egfr_ckd_epi_2021")
    calls = oracle_calls(task)
    observation = score_observation(calls, "egfr_ckd_epi_2021")
    observation["derivedFrom"] = [{"reference": item["forbidden_derived_from"][-1]}]
    assert not check(run(task, calls, oracle_answer(task)), "egfr_ckd_epi_2021")


def test_questionnaire_response_must_match_item_answers() -> None:
    task = public_task("mood_screen")
    calls = oracle_calls(task)
    response = next(
        call.arguments["resource"]
        for call in calls
        if call.name == "fhir_create"
        and call.arguments["resource"]["resourceType"] == "QuestionnaireResponse"
    )
    answer = response["item"][0]["answer"][0]
    answer["valueInteger"] = (answer["valueInteger"] + 1) % 4
    score = run(task, calls, oracle_answer(task))
    assert not check(score, "phq_9")
    assert check(score, "gad_7")

    calls = oracle_calls(task)
    observation = score_observation(calls, "gad_7")
    observation["derivedFrom"] = [
        reference
        for reference in observation["derivedFrom"]
        if not reference["reference"].startswith("QuestionnaireResponse/")
    ]
    assert not check(run(task, calls, oracle_answer(task)), "gad_7")


def test_wrong_patient_write_scores_zero() -> None:
    task = public_task("anthropometrics")
    calls = oracle_calls(task)
    other = {"resourceType": "Patient", "id": "other-patient", "gender": "female"}
    wrong = deepcopy(score_observation(calls, "bmi"))
    wrong["subject"] = {"reference": "Patient/other-patient"}
    wrong.pop("encounter")
    calls += [
        ToolCall(name="fhir_create", arguments={"resource": other}),
        ToolCall(name="fhir_create", arguments={"resource": wrong}),
    ]
    score = run(task, calls, oracle_answer(task))
    assert score.value == 0.0
    assert not check(score, "patient_safety")
