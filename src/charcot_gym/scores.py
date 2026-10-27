"""Deterministic clinical score calculators implemented from their primary publications."""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

LOINC_SYSTEM = "http://loinc.org"
SCORE_SYSTEM = "https://charcot.ai/fhir/CodeSystem/clinical-score"
BAND_SYSTEM = "https://charcot.ai/fhir/CodeSystem/clinical-score-band"

FieldType = Literal["number", "integer", "boolean", "enum"]


@dataclass(frozen=True)
class ScoreField:
    """One calculator input with its type, unit and allowed values."""

    name: str
    type: FieldType
    description: str
    unit: str | None = None
    allowed: tuple[Any, ...] | None = None
    minimum: float | None = None
    maximum: float | None = None
    precision: int | None = None
    conversions: tuple[tuple[str, float], ...] = ()

    def spec(self) -> dict[str, Any]:
        """Return the public field description."""
        data: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "unit": self.unit,
            "description": self.description,
        }
        if self.allowed is not None:
            data["allowed"] = list(self.allowed)
        if self.minimum is not None:
            data["minimum"] = self.minimum
        if self.maximum is not None:
            data["maximum"] = self.maximum
        if self.precision is not None:
            data["precision"] = self.precision
        if self.conversions:
            data["conversions"] = [
                {"from_unit": unit, "factor": factor} for unit, factor in self.conversions
            ]
        return data


@dataclass(frozen=True)
class ClinicalScore:
    """A score definition: inputs, formula, bands and FHIR code."""

    id: str
    name: str
    description: str
    citation: str
    fields: tuple[ScoreField, ...]
    compute: Callable[[Mapping[str, Any]], float]
    units: str | None
    code_system: str
    code: str
    display: str
    tolerance: float
    precision: int = 0
    band: Callable[[float, Mapping[str, Any]], str | None] | None = None
    questionnaire: str | None = None

    @property
    def integer(self) -> bool:
        """Whether the score is a whole-number point total."""
        return self.units is None and self.precision == 0

    def spec(self) -> dict[str, Any]:
        """Return the public score description."""
        return {
            "score": self.id,
            "name": self.name,
            "description": self.description,
            "citation": self.citation,
            "units": self.units,
            "code": {"system": self.code_system, "code": self.code, "display": self.display},
            "fields": [field.spec() for field in self.fields],
        }


class ScoreInputError(ValueError):
    """Structured calculator error listing missing and invalid fields."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        missing: list[str] | None = None,
        invalid: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.missing = missing or []
        self.invalid = invalid or []

    def to_dict(self) -> dict[str, Any]:
        """Return the error as tool output."""
        return {
            "code": self.code,
            "message": str(self),
            "missing": self.missing,
            "invalid": self.invalid,
        }


def _number(  # noqa: PLR0913
    name: str,
    unit: str,
    description: str,
    minimum: float = 0,
    *,
    precision: int | None = None,
    conversions: tuple[tuple[str, float], ...] = (),
) -> ScoreField:
    return ScoreField(
        name,
        "number",
        description,
        unit=unit,
        minimum=minimum,
        precision=precision,
        conversions=conversions,
    )


def _flag(name: str, description: str) -> ScoreField:
    return ScoreField(name, "boolean", description)


def _items(prefix: str, count: int, maximum: int, description: str) -> tuple[ScoreField, ...]:
    return tuple(
        ScoreField(
            f"{prefix}{index}",
            "integer",
            f"{description} item {index}",
            allowed=tuple(range(maximum + 1)),
        )
        for index in range(1, count + 1)
    )


AGE = ScoreField("age", "integer", "Age in completed years", unit="a", minimum=0, maximum=130)
SEX = ScoreField("sex", "enum", "Administrative sex", allowed=("female", "male"))
LB_TO_KG = 0.45359237
CREATININE_UMOL_PER_MG_DL = 88.4
BUN_MG_DL_PER_UREA_MMOL_L = 2.8
MG_DL_TO_MMOL_L = 0.02586
WEIGHT = _number("weight_kg", "kg", "Body weight", conversions=(("[lb_av]", LB_TO_KG),))
HEIGHT = _number("height_cm", "cm", "Body height", conversions=(("[in_i]", 2.54),))
CREATININE = _number(
    "creatinine_mg_dl",
    "mg/dL",
    "Serum creatinine",
    precision=2,
    conversions=(("umol/L", 1 / CREATININE_UMOL_PER_MG_DL),),
)
HEART_RATE = _number("heart_rate", "/min", "Heart rate")
RESPIRATORY_RATE = _number("respiratory_rate", "/min", "Respiratory rate")
SYSTOLIC = _number("systolic_bp", "mm[Hg]", "Systolic blood pressure")
DIASTOLIC = _number("diastolic_bp", "mm[Hg]", "Diastolic blood pressure")
AST = _number("ast", "U/L", "Aspartate aminotransferase")
ALT = _number("alt", "U/L", "Alanine aminotransferase")
PLATELETS = _number(
    "platelets", "10*9/L", "Platelet count", precision=0, conversions=(("10*3/uL", 1.0),)
)
SODIUM = _number("sodium", "mmol/L", "Serum sodium")


def _bands(value: float, cutoffs: tuple[tuple[float, str], ...], top: str) -> str:
    for upper, label in cutoffs:
        if value <= upper:
            return label
    return top


def _sum_flags(inputs: Mapping[str, Any], weights: Mapping[str, float]) -> float:
    return float(sum(weight for name, weight in weights.items() if inputs[name]))


def _bmi(i: Mapping[str, Any]) -> float:
    return float(i["weight_kg"] / (i["height_cm"] / 100) ** 2)


def _bmi_band(value: float, _: Mapping[str, Any]) -> str:
    if value < 18.5:
        return "underweight"
    if value < 25:
        return "normal"
    if value < 30:
        return "overweight"
    return "obese"


def _rr_seconds(i: Mapping[str, Any]) -> float:
    return float(60 / i["heart_rate"])


def _ckd_epi_2021(i: Mapping[str, Any]) -> float:
    female = i["sex"] == "female"
    ratio = i["creatinine_mg_dl"] / (0.7 if female else 0.9)
    alpha = -0.241 if female else -0.302
    value = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.2 * 0.9938 ** i["age"]
    return float(value * (1.012 if female else 1))


def _egfr_band(value: float, _: Mapping[str, Any]) -> str:
    if value >= 90:
        return "G1"
    if value >= 60:
        return "G2"
    if value >= 45:
        return "G3a"
    if value >= 30:
        return "G3b"
    if value >= 15:
        return "G4"
    return "G5"


def _cockcroft_gault(i: Mapping[str, Any]) -> float:
    value = (140 - i["age"]) * i["weight_kg"] / (72 * i["creatinine_mg_dl"])
    return float(value * (0.85 if i["sex"] == "female" else 1))


def _cha2ds2_vasc(i: Mapping[str, Any]) -> float:
    age = 2 if i["age"] >= 75 else 1 if i["age"] >= 65 else 0
    flags = _sum_flags(
        i,
        {
            "congestive_heart_failure": 1,
            "hypertension": 1,
            "diabetes": 1,
            "prior_stroke_tia_thromboembolism": 2,
            "vascular_disease": 1,
        },
    )
    return float(age + flags + (1 if i["sex"] == "female" else 0))


HAS_BLED_FLAGS = {
    "hypertension": 1,
    "abnormal_renal_function": 1,
    "abnormal_liver_function": 1,
    "prior_stroke": 1,
    "bleeding_history": 1,
    "labile_inr": 1,
    "elderly": 1,
    "antiplatelet_or_nsaid": 1,
    "alcohol_excess": 1,
}

WELLS_PE_FLAGS = {
    "clinical_signs_dvt": 3.0,
    "pe_most_likely_diagnosis": 3.0,
    "heart_rate_over_100": 1.5,
    "immobilization_or_surgery_4_weeks": 1.5,
    "previous_dvt_or_pe": 1.5,
    "hemoptysis": 1.0,
    "malignancy": 1.0,
}

WELLS_DVT_FLAGS = {
    "active_cancer": 1,
    "paralysis_or_recent_cast": 1,
    "bedridden_or_major_surgery": 1,
    "deep_vein_tenderness": 1,
    "entire_leg_swollen": 1,
    "calf_swelling_over_3_cm": 1,
    "pitting_edema_symptomatic_leg": 1,
    "collateral_superficial_veins": 1,
    "previous_dvt": 1,
    "alternative_diagnosis_as_likely": -2,
}

PERC_FLAGS = {
    "unilateral_leg_swelling": 1,
    "hemoptysis": 1,
    "recent_surgery_or_trauma": 1,
    "prior_dvt_or_pe": 1,
    "hormone_use": 1,
}

GENEVA_FLAGS = {
    "previous_dvt_or_pe": 3,
    "surgery_or_fracture_1_month": 2,
    "active_malignancy": 2,
    "unilateral_lower_limb_pain": 3,
    "hemoptysis": 2,
    "deep_vein_palpation_pain_and_edema": 4,
}

TIMI_FLAGS = {
    "three_or_more_cad_risk_factors": 1,
    "known_cad_stenosis_50": 1,
    "aspirin_last_7_days": 1,
    "severe_angina_2_episodes_24h": 1,
    "st_deviation_0_5_mm": 1,
    "positive_cardiac_marker": 1,
}

CENTOR_FLAGS = {
    "tonsillar_exudate": 1,
    "tender_anterior_cervical_nodes": 1,
    "fever_history": 1,
    "cough_absent": 1,
}


def _perc(i: Mapping[str, Any]) -> float:
    return float(
        (i["age"] >= 50)
        + (i["heart_rate"] >= 100)
        + (i["oxygen_saturation"] < 95)
        + _sum_flags(i, PERC_FLAGS)
    )


def _revised_geneva(i: Mapping[str, Any]) -> float:
    heart = 5 if i["heart_rate"] >= 95 else 3 if i["heart_rate"] >= 75 else 0
    return float((1 if i["age"] > 65 else 0) + heart + _sum_flags(i, GENEVA_FLAGS))


def _heart(i: Mapping[str, Any]) -> float:
    age = 2 if i["age"] >= 65 else 1 if i["age"] >= 45 else 0
    return float(age + i["history"] + i["ecg"] + i["risk_factors"] + i["troponin"])


def _curb_65(i: Mapping[str, Any]) -> float:
    return float(
        bool(i["confusion"])
        + (i["urea_mmol_l"] > 7)
        + (i["respiratory_rate"] >= 30)
        + (i["systolic_bp"] < 90 or i["diastolic_bp"] <= 60)
        + (i["age"] >= 65)
    )


def _qsofa(i: Mapping[str, Any]) -> float:
    return float(
        (i["respiratory_rate"] >= 22) + bool(i["altered_mentation"]) + (i["systolic_bp"] <= 100)
    )


def _child_pugh(i: Mapping[str, Any]) -> float:
    bilirubin = 1 if i["bilirubin_mg_dl"] < 2 else 2 if i["bilirubin_mg_dl"] <= 3 else 3
    albumin = 1 if i["albumin_g_dl"] > 3.5 else 2 if i["albumin_g_dl"] >= 2.8 else 3
    prothrombin = (
        1
        if i["prothrombin_prolongation_s"] < 4
        else 2
        if i["prothrombin_prolongation_s"] <= 6
        else 3
    )
    grades = {"none": 1, "slight": 2, "moderate": 3}
    encephalopathy = {"none": 1, "grade_1_2": 2, "grade_3_4": 3}
    return float(
        bilirubin
        + albumin
        + prothrombin
        + grades[i["ascites"]]
        + encephalopathy[i["encephalopathy"]]
    )


def _meld(i: Mapping[str, Any]) -> float:
    creatinine = 4.0 if i["dialysis_twice_past_week"] else min(i["creatinine_mg_dl"], 4.0)
    return float(
        3.78 * math.log(max(i["bilirubin_mg_dl"], 1.0))
        + 11.2 * math.log(max(i["inr"], 1.0))
        + 9.57 * math.log(max(creatinine, 1.0))
        + 6.43
    )


def _meld_na(i: Mapping[str, Any]) -> float:
    meld = _meld(i)
    sodium = min(max(i["sodium"], 125.0), 140.0)
    return float(meld - sodium - 0.025 * meld * (140 - sodium) + 140)


def _abcd2(i: Mapping[str, Any]) -> float:
    clinical = {"unilateral_weakness": 2, "speech_only": 1, "other": 0}[i["clinical_features"]]
    duration = 2 if i["duration_min"] >= 60 else 1 if i["duration_min"] >= 10 else 0
    return float(
        (i["age"] >= 60)
        + (i["systolic_bp"] >= 140 or i["diastolic_bp"] >= 90)
        + clinical
        + duration
        + bool(i["diabetes"])
    )


def _mcisaac(i: Mapping[str, Any]) -> float:
    age = 1 if 3 <= i["age"] <= 14 else -1 if i["age"] >= 45 else 0
    return _sum_flags(i, CENTOR_FLAGS) + age


def _total(prefix: str, count: int) -> Callable[[Mapping[str, Any]], float]:
    return lambda i: float(sum(i[f"{prefix}{index}"] for index in range(1, count + 1)))


def _points_band(cutoffs: tuple[tuple[float, str], ...], top: str) -> Callable[..., str]:
    return lambda value, _: _bands(value, cutoffs, top)


def _flags(weights: Mapping[str, float], descriptions: Mapping[str, str]) -> tuple[ScoreField, ...]:
    return tuple(_flag(name, descriptions[name]) for name in weights)


SCORES: dict[str, ClinicalScore] = {}


def _register(score: ClinicalScore) -> None:
    SCORES[score.id] = score


_register(
    ClinicalScore(
        id="bmi",
        name="Body mass index",
        description="Weight divided by the square of height, with the WHO weight classes.",
        citation=(
            "Keys A, et al. Indices of relative weight and obesity. J Chronic Dis. "
            "1972;25:329-343. doi:10.1016/0021-9681(72)90027-6. Classes: WHO Technical "
            "Report Series 894, 2000."
        ),
        fields=(WEIGHT, HEIGHT),
        compute=_bmi,
        units="kg/m2",
        code_system=LOINC_SYSTEM,
        code="39156-5",
        display="Body mass index (BMI) [Ratio]",
        tolerance=0.1,
        precision=1,
        band=_bmi_band,
    )
)
_register(
    ClinicalScore(
        id="bsa_mosteller",
        name="Body surface area (Mosteller)",
        description="Square root of height times weight divided by 3600.",
        citation=(
            "Mosteller RD. Simplified calculation of body-surface area. N Engl J Med. "
            "1987;317:1098. doi:10.1056/NEJM198710223171717."
        ),
        fields=(HEIGHT, WEIGHT),
        compute=lambda i: float(math.sqrt(i["height_cm"] * i["weight_kg"] / 3600)),
        units="m2",
        code_system=LOINC_SYSTEM,
        code="3140-1",
        display="Body surface area Derived from formula",
        tolerance=0.01,
        precision=2,
    )
)
QT = _number("qt_ms", "ms", "Measured QT interval")
_register(
    ClinicalScore(
        id="qtc_bazett",
        name="Corrected QT interval (Bazett)",
        description="QT divided by the square root of the RR interval in seconds.",
        citation=(
            "Bazett HC. An analysis of the time-relations of electrocardiograms. "
            "Heart. 1920;7:353-370."
        ),
        fields=(QT, HEART_RATE),
        compute=lambda i: float(i["qt_ms"] / math.sqrt(_rr_seconds(i))),
        units="ms",
        code_system=LOINC_SYSTEM,
        code="76635-2",
        display="Q-T interval corrected based on Bazett formula",
        tolerance=2,
    )
)
_register(
    ClinicalScore(
        id="qtc_fridericia",
        name="Corrected QT interval (Fridericia)",
        description="QT divided by the cube root of the RR interval in seconds.",
        citation=(
            "Fridericia LS. Die Systolendauer im Elektrokardiogramm bei normalen Menschen "
            "und bei Herzkranken. Acta Med Scand. 1920;53:469-486. "
            "doi:10.1111/j.0954-6820.1920.tb18266.x."
        ),
        fields=(QT, HEART_RATE),
        compute=lambda i: float(i["qt_ms"] / _rr_seconds(i) ** (1 / 3)),
        units="ms",
        code_system=LOINC_SYSTEM,
        code="76634-5",
        display="Q-T interval corrected based on Fridericia formula",
        tolerance=2,
    )
)
CHLORIDE = _number("chloride", "mmol/L", "Serum chloride")
BICARBONATE = _number("bicarbonate", "mmol/L", "Serum bicarbonate")
_register(
    ClinicalScore(
        id="anion_gap",
        name="Serum anion gap",
        description="Sodium minus the sum of chloride and bicarbonate.",
        citation=(
            "Emmett M, Narins RG. Clinical use of the anion gap. Medicine (Baltimore). "
            "1977;56:38-54. doi:10.1097/00005792-197701000-00002."
        ),
        fields=(SODIUM, CHLORIDE, BICARBONATE),
        compute=lambda i: float(i["sodium"] - i["chloride"] - i["bicarbonate"]),
        units="mmol/L",
        code_system=LOINC_SYSTEM,
        code="33037-3",
        display="Anion gap in Serum or Plasma by calculation",
        tolerance=0.5,
        precision=1,
    )
)
ALBUMIN = _number("albumin_g_dl", "g/dL", "Serum albumin", conversions=(("g/L", 0.1),))
_register(
    ClinicalScore(
        id="corrected_calcium",
        name="Albumin-adjusted calcium",
        description="Total calcium adjusted by 0.8 mg/dL for each g/dL of albumin below 4.",
        citation=(
            "Payne RB, et al. Interpretation of serum calcium in patients with abnormal "
            "serum proteins. Br Med J. 1973;4:643-646. doi:10.1136/bmj.4.5893.643."
        ),
        fields=(_number("calcium_mg_dl", "mg/dL", "Total serum calcium"), ALBUMIN),
        compute=lambda i: float(i["calcium_mg_dl"] + 0.8 * (4 - i["albumin_g_dl"])),
        units="mg/dL",
        code_system=LOINC_SYSTEM,
        code="46099-8",
        display="Calcium [Mass/volume] corrected for albumin in Serum or Plasma",
        tolerance=0.1,
        precision=1,
    )
)
_register(
    ClinicalScore(
        id="corrected_sodium_katz",
        name="Glucose-corrected sodium (Katz)",
        description="Sodium raised 1.6 mmol/L for each 100 mg/dL of glucose above 100 mg/dL.",
        citation=(
            "Katz MA. Hyperglycemia-induced hyponatremia: calculation of expected serum "
            "sodium depression. N Engl J Med. 1973;289:843-844. "
            "doi:10.1056/NEJM197310182891607."
        ),
        fields=(SODIUM, _number("glucose_mg_dl", "mg/dL", "Serum glucose")),
        compute=lambda i: float(i["sodium"] + 1.6 * max(i["glucose_mg_dl"] - 100, 0) / 100),
        units="mmol/L",
        code_system=SCORE_SYSTEM,
        code="corrected_sodium_katz",
        display="Glucose-corrected sodium (Katz)",
        tolerance=0.5,
        precision=1,
    )
)
_register(
    ClinicalScore(
        id="egfr_ckd_epi_2021",
        name="eGFR (CKD-EPI 2021 creatinine)",
        description="Race-free creatinine equation, with the KDIGO GFR categories.",
        citation=(
            "Inker LA, et al. New creatinine- and cystatin C-based equations to estimate "
            "GFR without race. N Engl J Med. 2021;385:1737-1749. doi:10.1056/NEJMoa2102953. "
            "Categories: KDIGO 2012 CKD guideline, Kidney Int Suppl. 2013;3:1-150."
        ),
        fields=(AGE, SEX, CREATININE),
        compute=_ckd_epi_2021,
        units="mL/min/{1.73_m2}",
        code_system=LOINC_SYSTEM,
        code="98979-8",
        display="Glomerular filtration rate by Creatinine-based formula (CKD-EPI 2021)",
        tolerance=1,
        band=_egfr_band,
    )
)
_register(
    ClinicalScore(
        id="crcl_cockcroft_gault",
        name="Creatinine clearance (Cockcroft-Gault)",
        description="(140 - age) times weight over 72 times creatinine, times 0.85 if female.",
        citation=(
            "Cockcroft DW, Gault MH. Prediction of creatinine clearance from serum "
            "creatinine. Nephron. 1976;16:31-41. doi:10.1159/000180580."
        ),
        fields=(AGE, SEX, WEIGHT, CREATININE),
        compute=_cockcroft_gault,
        units="mL/min",
        code_system=LOINC_SYSTEM,
        code="35591-7",
        display="Creatinine renal clearance predicted by Cockcroft-Gault formula",
        tolerance=1,
    )
)
_register(
    ClinicalScore(
        id="cha2ds2_vasc",
        name="CHA2DS2-VASc",
        description="Stroke risk points in atrial fibrillation.",
        citation=(
            "Lip GY, et al. Refining clinical risk stratification for predicting stroke and "
            "thromboembolism in atrial fibrillation using a novel risk factor-based approach. "
            "Chest. 2010;137:263-272. doi:10.1378/chest.09-1584."
        ),
        fields=(
            AGE,
            SEX,
            _flag("congestive_heart_failure", "Heart failure or moderate-severe LV dysfunction"),
            _flag("hypertension", "Hypertension"),
            _flag("diabetes", "Diabetes mellitus"),
            _flag("prior_stroke_tia_thromboembolism", "Prior stroke, TIA or thromboembolism"),
            _flag("vascular_disease", "Prior myocardial infarction, peripheral or aortic plaque"),
        ),
        compute=_cha2ds2_vasc,
        units=None,
        code_system=SCORE_SYSTEM,
        code="cha2ds2_vasc",
        display="CHA2DS2-VASc score",
        tolerance=0,
        band=_points_band(((0, "low"), (1, "intermediate")), "high"),
    )
)
_register(
    ClinicalScore(
        id="has_bled",
        name="HAS-BLED",
        description="Major bleeding risk points in anticoagulated atrial fibrillation.",
        citation=(
            "Pisters R, et al. A novel user-friendly score (HAS-BLED) to assess 1-year risk "
            "of major bleeding in patients with atrial fibrillation. Chest. "
            "2010;138:1093-1100. doi:10.1378/chest.10-0134."
        ),
        fields=_flags(
            HAS_BLED_FLAGS,
            {
                "hypertension": "Uncontrolled hypertension, systolic above 160 mm Hg",
                "abnormal_renal_function": "Dialysis, transplant or creatinine >= 200 umol/L",
                "abnormal_liver_function": "Cirrhosis or bilirubin > 2x and AST/ALT > 3x normal",
                "prior_stroke": "Prior stroke",
                "bleeding_history": "Prior bleeding or predisposition",
                "labile_inr": "Labile INR",
                "elderly": "Age above 65 years",
                "antiplatelet_or_nsaid": "Concomitant antiplatelet agent or NSAID",
                "alcohol_excess": "Alcohol excess",
            },
        ),
        compute=lambda i: _sum_flags(i, HAS_BLED_FLAGS),
        units=None,
        code_system=SCORE_SYSTEM,
        code="has_bled",
        display="HAS-BLED score",
        tolerance=0,
        band=_points_band(((2, "not_high"),), "high"),
    )
)
_register(
    ClinicalScore(
        id="wells_pe",
        name="Wells score for pulmonary embolism",
        description="Pretest probability points for pulmonary embolism, three-level model.",
        citation=(
            "Wells PS, et al. Derivation of a simple clinical model to categorize patients "
            "probability of pulmonary embolism. Thromb Haemost. 2000;83:416-420. "
            "doi:10.1055/s-0037-1613830."
        ),
        fields=_flags(
            WELLS_PE_FLAGS,
            {
                "clinical_signs_dvt": "Clinical signs and symptoms of DVT",
                "pe_most_likely_diagnosis": "PE as or more likely than an alternative diagnosis",
                "heart_rate_over_100": "Heart rate above 100/min",
                "immobilization_or_surgery_4_weeks": "Immobilization or surgery in past 4 weeks",
                "previous_dvt_or_pe": "Previous DVT or PE",
                "hemoptysis": "Hemoptysis",
                "malignancy": "Malignancy on treatment or within 6 months or palliative",
            },
        ),
        compute=lambda i: _sum_flags(i, WELLS_PE_FLAGS),
        units=None,
        code_system=LOINC_SYSTEM,
        code="89547-4",
        display="Total score Wells pulmonary embolism risk calculator",
        tolerance=0,
        precision=1,
        band=_points_band(((1.5, "low"), (6, "moderate")), "high"),
    )
)
_register(
    ClinicalScore(
        id="wells_dvt",
        name="Wells score for deep vein thrombosis",
        description="Pretest probability points for DVT, two-level model.",
        citation=(
            "Wells PS, et al. Evaluation of D-dimer in the diagnosis of suspected deep-vein "
            "thrombosis. N Engl J Med. 2003;349:1227-1235. doi:10.1056/NEJMoa023153."
        ),
        fields=_flags(
            WELLS_DVT_FLAGS,
            {
                "active_cancer": "Active cancer",
                "paralysis_or_recent_cast": "Paralysis, paresis or recent leg cast",
                "bedridden_or_major_surgery": "Bedridden 3 days or major surgery in 12 weeks",
                "deep_vein_tenderness": "Localized tenderness along the deep veins",
                "entire_leg_swollen": "Entire leg swollen",
                "calf_swelling_over_3_cm": "Calf swelling more than 3 cm versus the other leg",
                "pitting_edema_symptomatic_leg": "Pitting edema confined to the symptomatic leg",
                "collateral_superficial_veins": "Collateral nonvaricose superficial veins",
                "previous_dvt": "Previously documented DVT",
                "alternative_diagnosis_as_likely": "Alternative diagnosis at least as likely",
            },
        ),
        compute=lambda i: _sum_flags(i, WELLS_DVT_FLAGS),
        units=None,
        code_system=SCORE_SYSTEM,
        code="wells_dvt",
        display="Wells DVT score",
        tolerance=0,
        band=_points_band(((1, "unlikely"),), "likely"),
    )
)
OXYGEN = _number("oxygen_saturation", "%", "Oxygen saturation on room air")
_register(
    ClinicalScore(
        id="perc",
        name="Pulmonary embolism rule-out criteria",
        description="Count of positive PERC criteria; zero means the rule is negative.",
        citation=(
            "Kline JA, et al. Clinical criteria to prevent unnecessary diagnostic testing in "
            "emergency department patients with suspected pulmonary embolism. J Thromb "
            "Haemost. 2004;2:1247-1255. doi:10.1111/j.1538-7836.2004.00790.x."
        ),
        fields=(
            AGE,
            HEART_RATE,
            OXYGEN,
            *_flags(
                PERC_FLAGS,
                {
                    "unilateral_leg_swelling": "Unilateral leg swelling",
                    "hemoptysis": "Hemoptysis",
                    "recent_surgery_or_trauma": "Surgery or trauma in the past 4 weeks",
                    "prior_dvt_or_pe": "Prior DVT or PE",
                    "hormone_use": "Exogenous estrogen use",
                },
            ),
        ),
        compute=_perc,
        units=None,
        code_system=LOINC_SYSTEM,
        code="89545-8",
        display="PERC Total score",
        tolerance=0,
        band=_points_band(((0, "negative"),), "positive"),
    )
)
_register(
    ClinicalScore(
        id="revised_geneva",
        name="Revised Geneva score",
        description="Pretest probability points for pulmonary embolism without clinician gestalt.",
        citation=(
            "Le Gal G, et al. Prediction of pulmonary embolism in the emergency department: "
            "the revised Geneva score. Ann Intern Med. 2006;144:165-171. "
            "doi:10.7326/0003-4819-144-3-200602070-00004."
        ),
        fields=(
            AGE,
            HEART_RATE,
            *_flags(
                GENEVA_FLAGS,
                {
                    "previous_dvt_or_pe": "Previous DVT or PE",
                    "surgery_or_fracture_1_month": "Surgery or lower limb fracture within 1 month",
                    "active_malignancy": "Active solid or hematologic malignancy",
                    "unilateral_lower_limb_pain": "Unilateral lower limb pain",
                    "hemoptysis": "Hemoptysis",
                    "deep_vein_palpation_pain_and_edema": (
                        "Pain on lower limb deep venous palpation and unilateral edema"
                    ),
                },
            ),
        ),
        compute=_revised_geneva,
        units=None,
        code_system=LOINC_SYSTEM,
        code="89548-2",
        display="Total score Revised Geneva Score",
        tolerance=0,
        band=_points_band(((3, "low"), (10, "intermediate")), "high"),
    )
)
COMPONENT = (0, 1, 2)
_register(
    ClinicalScore(
        id="heart",
        name="HEART score",
        description="Chest pain risk points from history, ECG, age, risk factors and troponin.",
        citation=(
            "Six AJ, Backus BE, Kelder JC. Chest pain in the emergency room: value of the "
            "HEART score. Neth Heart J. 2008;16:191-196. doi:10.1007/BF03086144."
        ),
        fields=(
            ScoreField(
                "history",
                "integer",
                "History: 0 slightly, 1 moderately, 2 highly suspicious",
                allowed=COMPONENT,
            ),
            ScoreField(
                "ecg",
                "integer",
                "ECG: 0 normal, 1 nonspecific repolarization, 2 significant ST deviation",
                allowed=COMPONENT,
            ),
            AGE,
            ScoreField(
                "risk_factors",
                "integer",
                "Risk factors: 0 none, 1 one or two, 2 three or more or known atherosclerosis",
                allowed=COMPONENT,
            ),
            ScoreField(
                "troponin",
                "integer",
                "Troponin: 0 normal, 1 one to three times, 2 above three times the limit",
                allowed=COMPONENT,
            ),
        ),
        compute=_heart,
        units=None,
        code_system=SCORE_SYSTEM,
        code="heart",
        display="HEART score",
        tolerance=0,
        band=_points_band(((3, "low"), (6, "moderate")), "high"),
    )
)
_register(
    ClinicalScore(
        id="timi_ua_nstemi",
        name="TIMI risk score for UA/NSTEMI",
        description="Count of seven risk predictors in unstable angina or non-ST elevation MI.",
        citation=(
            "Antman EM, et al. The TIMI risk score for unstable angina/non-ST elevation MI. "
            "JAMA. 2000;284:835-842. doi:10.1001/jama.284.7.835."
        ),
        fields=(
            AGE,
            *_flags(
                TIMI_FLAGS,
                {
                    "three_or_more_cad_risk_factors": "At least three coronary risk factors",
                    "known_cad_stenosis_50": "Known coronary stenosis of at least 50%",
                    "aspirin_last_7_days": "Aspirin use in the past 7 days",
                    "severe_angina_2_episodes_24h": "At least two angina episodes in 24 hours",
                    "st_deviation_0_5_mm": "ST deviation of at least 0.5 mm",
                    "positive_cardiac_marker": "Elevated serum cardiac marker",
                },
            ),
        ),
        compute=lambda i: float((i["age"] >= 65) + _sum_flags(i, TIMI_FLAGS)),
        units=None,
        code_system=SCORE_SYSTEM,
        code="timi_ua_nstemi",
        display="TIMI risk score for UA/NSTEMI",
        tolerance=0,
    )
)
_register(
    ClinicalScore(
        id="curb_65",
        name="CURB-65",
        description="Community-acquired pneumonia severity points.",
        citation=(
            "Lim WS, et al. Defining community acquired pneumonia severity on presentation "
            "to hospital: an international derivation and validation study. Thorax. "
            "2003;58:377-382. doi:10.1136/thorax.58.5.377."
        ),
        fields=(
            _flag("confusion", "New mental confusion"),
            _number(
                "urea_mmol_l",
                "mmol/L",
                "Serum urea; urea nitrogen in mg/dL converts with the factor",
                conversions=(("mg/dL", 1 / BUN_MG_DL_PER_UREA_MMOL_L),),
            ),
            RESPIRATORY_RATE,
            SYSTOLIC,
            DIASTOLIC,
            AGE,
        ),
        compute=_curb_65,
        units=None,
        code_system=SCORE_SYSTEM,
        code="curb_65",
        display="CURB-65 score",
        tolerance=0,
        band=_points_band(((1, "low"), (2, "intermediate")), "high"),
    )
)
_register(
    ClinicalScore(
        id="qsofa",
        name="Quick SOFA",
        description="Bedside sepsis screen; two or more points is positive.",
        citation=(
            "Seymour CW, et al. Assessment of clinical criteria for sepsis: for the Third "
            "International Consensus Definitions for Sepsis and Septic Shock (Sepsis-3). "
            "JAMA. 2016;315:762-774. doi:10.1001/jama.2016.0288."
        ),
        fields=(RESPIRATORY_RATE, _flag("altered_mentation", "Altered mentation"), SYSTOLIC),
        compute=_qsofa,
        units=None,
        code_system=LOINC_SYSTEM,
        code="96791-9",
        display="Quick SOFA score SOFA.quick",
        tolerance=0,
        band=_points_band(((1, "negative"),), "positive"),
    )
)
BILIRUBIN = _number("bilirubin_mg_dl", "mg/dL", "Total serum bilirubin")
_register(
    ClinicalScore(
        id="child_pugh",
        name="Child-Pugh score",
        description=(
            "Cirrhosis severity from bilirubin, albumin, prothrombin time, ascites and "
            "encephalopathy."
        ),
        citation=(
            "Pugh RN, et al. Transection of the oesophagus for bleeding oesophageal varices. "
            "Br J Surg. 1973;60:646-649. doi:10.1002/bjs.1800600817."
        ),
        fields=(
            BILIRUBIN,
            ALBUMIN,
            _number(
                "prothrombin_prolongation_s", "s", "Prothrombin time prolongation over control"
            ),
            ScoreField("ascites", "enum", "Ascites", allowed=("none", "slight", "moderate")),
            ScoreField(
                "encephalopathy",
                "enum",
                "Encephalopathy",
                allowed=("none", "grade_1_2", "grade_3_4"),
            ),
        ),
        compute=_child_pugh,
        units=None,
        code_system=LOINC_SYSTEM,
        code="98152-2",
        display="Total score [Child-Pugh]",
        tolerance=0,
        band=_points_band(((6, "A"), (9, "B")), "C"),
    )
)
_register(
    ClinicalScore(
        id="meld_na",
        name="MELD-Na",
        description="MELD with serum sodium, sodium bounded to 125-140 mmol/L.",
        citation=(
            "Kim WR, et al. Hyponatremia and mortality among patients on the liver-"
            "transplant waiting list. N Engl J Med. 2008;359:1018-1026. "
            "doi:10.1056/NEJMoa0801209. MELD: Kamath PS, et al. Hepatology. "
            "2001;33:464-470. doi:10.1053/jhep.2001.22172."
        ),
        fields=(
            BILIRUBIN,
            _number("inr", "{INR}", "International normalized ratio"),
            CREATININE,
            SODIUM,
            _flag("dialysis_twice_past_week", "Dialysis at least twice in the past week"),
        ),
        compute=_meld_na,
        units=None,
        code_system=LOINC_SYSTEM,
        code="100348-2",
        display="Model for end-stage liver disease sodium score",
        tolerance=0.5,
        precision=1,
    )
)
_register(
    ClinicalScore(
        id="fib_4",
        name="FIB-4 index",
        description="Age times AST over platelets times the square root of ALT.",
        citation=(
            "Sterling RK, et al. Development of a simple noninvasive index to predict "
            "significant fibrosis in patients with HIV/HCV coinfection. Hepatology. "
            "2006;43:1317-1325. doi:10.1002/hep.21178."
        ),
        fields=(AGE, AST, ALT, PLATELETS),
        compute=lambda i: float(i["age"] * i["ast"] / (i["platelets"] * math.sqrt(i["alt"]))),
        units=None,
        code_system=LOINC_SYSTEM,
        code="98488-0",
        display="Liver fibrosis score in Serum Calculated by FIB4",
        tolerance=0.05,
        precision=2,
        band=lambda value, _: (
            "low" if value < 1.45 else "high" if value > 3.25 else "indeterminate"
        ),
    )
)
_register(
    ClinicalScore(
        id="apri",
        name="AST to platelet ratio index",
        description="AST over its upper limit of normal, times 100, over platelets.",
        citation=(
            "Wai CT, et al. A simple noninvasive index can predict both significant fibrosis "
            "and cirrhosis in patients with chronic hepatitis C. Hepatology. "
            "2003;38:518-526. doi:10.1053/jhep.2003.50346."
        ),
        fields=(
            AST,
            _number("ast_upper_limit", "U/L", "Laboratory upper limit of normal for AST"),
            PLATELETS,
        ),
        compute=lambda i: float(i["ast"] / i["ast_upper_limit"] * 100 / i["platelets"]),
        units=None,
        code_system=SCORE_SYSTEM,
        code="apri",
        display="AST to platelet ratio index",
        tolerance=0.05,
        precision=2,
        band=lambda value, _: (
            "low" if value <= 0.5 else "high" if value >= 1.5 else "indeterminate"
        ),
    )
)
_register(
    ClinicalScore(
        id="abcd2",
        name="ABCD2 score",
        description="Early stroke risk points after transient ischemic attack.",
        citation=(
            "Johnston SC, et al. Validation and refinement of scores to predict very early "
            "stroke risk after transient ischaemic attack. Lancet. 2007;369:283-292. "
            "doi:10.1016/S0140-6736(07)60150-0."
        ),
        fields=(
            AGE,
            SYSTOLIC,
            DIASTOLIC,
            ScoreField(
                "clinical_features",
                "enum",
                "Unilateral weakness, speech disturbance without weakness, or other",
                allowed=("unilateral_weakness", "speech_only", "other"),
            ),
            _number("duration_min", "min", "Symptom duration"),
            _flag("diabetes", "Diabetes mellitus"),
        ),
        compute=_abcd2,
        units=None,
        code_system=SCORE_SYSTEM,
        code="abcd2",
        display="ABCD2 score",
        tolerance=0,
        band=_points_band(((3, "low"), (5, "moderate")), "high"),
    )
)
CENTOR_DESCRIPTIONS = {
    "tonsillar_exudate": "Tonsillar exudate or swelling",
    "tender_anterior_cervical_nodes": "Tender or swollen anterior cervical nodes",
    "fever_history": "History of fever or temperature above 38 C",
    "cough_absent": "Absence of cough",
}
_register(
    ClinicalScore(
        id="centor",
        name="Centor criteria",
        description="Count of four clinical predictors of streptococcal pharyngitis in adults.",
        citation=(
            "Centor RM, et al. The diagnosis of strep throat in adults in the emergency room. "
            "Med Decis Making. 1981;1:239-246. doi:10.1177/0272989X8100100304."
        ),
        fields=_flags(CENTOR_FLAGS, CENTOR_DESCRIPTIONS),
        compute=lambda i: _sum_flags(i, CENTOR_FLAGS),
        units=None,
        code_system=SCORE_SYSTEM,
        code="centor",
        display="Centor score",
        tolerance=0,
    )
)
_register(
    ClinicalScore(
        id="mcisaac",
        name="McIsaac score",
        description="Centor criteria plus one point for age 3-14, minus one for age 45 or older.",
        citation=(
            "McIsaac WJ, et al. A clinical score to reduce unnecessary antibiotic use in "
            "patients with sore throat. CMAJ. 1998;158:75-83."
        ),
        fields=(AGE, *_flags(CENTOR_FLAGS, CENTOR_DESCRIPTIONS)),
        compute=_mcisaac,
        units=None,
        code_system=SCORE_SYSTEM,
        code="mcisaac",
        display="McIsaac score",
        tolerance=0,
    )
)
_register(
    ClinicalScore(
        id="gcs",
        name="Glasgow Coma Scale",
        description="Sum of eye opening (1-4), verbal (1-5) and motor (1-6) responses.",
        citation=(
            "Teasdale G, Jennett B. Assessment of coma and impaired consciousness. A practical "
            "scale. Lancet. 1974;2:81-84. doi:10.1016/S0140-6736(74)91639-0. Six-level motor "
            "scale: Teasdale G, Jennett B. Assessment and prognosis of coma after head injury. "
            "Acta Neurochir. 1976;34:45-55. doi:10.1007/BF01405862."
        ),
        fields=(
            ScoreField("eye", "integer", "Best eye opening response", allowed=(1, 2, 3, 4)),
            ScoreField("verbal", "integer", "Best verbal response", allowed=(1, 2, 3, 4, 5)),
            ScoreField("motor", "integer", "Best motor response", allowed=(1, 2, 3, 4, 5, 6)),
        ),
        compute=lambda i: float(i["eye"] + i["verbal"] + i["motor"]),
        units=None,
        code_system=LOINC_SYSTEM,
        code="9269-2",
        display="Glasgow coma score total",
        tolerance=0,
    )
)
_register(
    ClinicalScore(
        id="phq_9",
        name="PHQ-9",
        description="Sum of nine depression items scored 0-3, with severity bands.",
        citation=(
            "Kroenke K, Spitzer RL, Williams JB. The PHQ-9: validity of a brief depression "
            "severity measure. J Gen Intern Med. 2001;16:606-613. "
            "doi:10.1046/j.1525-1497.2001.016009606.x."
        ),
        fields=_items("item_", 9, 3, "PHQ-9"),
        compute=_total("item_", 9),
        units=None,
        code_system=LOINC_SYSTEM,
        code="44261-6",
        display="Patient Health Questionnaire 9 item (PHQ-9) total score [Reported]",
        tolerance=0,
        band=_points_band(
            ((4, "minimal"), (9, "mild"), (14, "moderate"), (19, "moderately_severe")), "severe"
        ),
        questionnaire="http://loinc.org/q/44249-1",
    )
)
_register(
    ClinicalScore(
        id="gad_7",
        name="GAD-7",
        description="Sum of seven anxiety items scored 0-3, with severity bands.",
        citation=(
            "Spitzer RL, et al. A brief measure for assessing generalized anxiety disorder: "
            "the GAD-7. Arch Intern Med. 2006;166:1092-1097. doi:10.1001/archinte.166.10.1092."
        ),
        fields=_items("item_", 7, 3, "GAD-7"),
        compute=_total("item_", 7),
        units=None,
        code_system=LOINC_SYSTEM,
        code="70274-6",
        display="Generalized anxiety disorder 7 item (GAD-7) total score [Reported.PHQ]",
        tolerance=0,
        band=_points_band(((4, "minimal"), (9, "mild"), (14, "moderate")), "severe"),
        questionnaire="http://loinc.org/q/69737-5",
    )
)
_register(
    ClinicalScore(
        id="audit_c",
        name="AUDIT-C",
        description="Sum of the three alcohol consumption items scored 0-4.",
        citation=(
            "Bush K, et al. The AUDIT alcohol consumption questions (AUDIT-C): an effective "
            "brief screening test for problem drinking. Arch Intern Med. 1998;158:1789-1795. "
            "doi:10.1001/archinte.158.16.1789."
        ),
        fields=_items("item_", 3, 4, "AUDIT-C"),
        compute=_total("item_", 3),
        units=None,
        code_system=LOINC_SYSTEM,
        code="75626-2",
        display="Total score [AUDIT-C]",
        tolerance=0,
        questionnaire="http://loinc.org/q/72109-2",
    )
)


CHOLESTEROL_CONVERSIONS = (("mmol/L", 1 / MG_DL_TO_MMOL_L),)
TOTAL_CHOLESTEROL = _number(
    "total_cholesterol_mg_dl", "mg/dL", "Total cholesterol", conversions=CHOLESTEROL_CONVERSIONS
)
HDL_CHOLESTEROL = _number(
    "hdl_mg_dl", "mg/dL", "HDL cholesterol", conversions=CHOLESTEROL_CONVERSIONS
)
RISK_DIABETES = _flag("diabetes", "Diabetes mellitus")
CURRENT_SMOKER = _flag("current_smoker", "Current cigarette smoking")
ANTIHYPERTENSIVE = _flag("antihypertensive", "Current antihypertensive medication")

# Simplified base model for 10-year total CVD, in this term order: age, non-HDL-C, HDL-C,
# SBP <110, SBP >=110, diabetes, smoking, eGFR <60, eGFR >=60, antihypertensive, statin,
# treated SBP, treated non-HDL-C, age x non-HDL-C, age x HDL-C, age x SBP, age x diabetes,
# age x smoking, age x eGFR <60, constant.
PREVENT_TOTAL_CVD: dict[str, tuple[float, ...]] = {
    "female": (
        0.7939329,
        0.0305239,
        -0.1606857,
        -0.2394003,
        0.3600781,
        0.8667604,
        0.5360739,
        0.6045917,
        0.0433769,
        0.3151672,
        -0.1477655,
        -0.0663612,
        0.1197879,
        -0.0819715,
        0.0306769,
        -0.0946348,
        -0.27057,
        -0.078715,
        -0.1637806,
        -3.307728,
    ),
    "male": (
        0.7688528,
        0.0736174,
        -0.0954431,
        -0.4347345,
        0.3362658,
        0.7692857,
        0.4386871,
        0.5378979,
        0.0164827,
        0.288879,
        -0.1337349,
        -0.0475924,
        0.150273,
        -0.0517874,
        0.0191169,
        -0.1049477,
        -0.2251948,
        -0.0895067,
        -0.1543702,
        -3.031168,
    ),
}


def _prevent_cvd_10y(i: Mapping[str, Any]) -> float:
    age = (i["age"] - 55) / 10
    non_hdl = (i["total_cholesterol_mg_dl"] - i["hdl_mg_dl"]) * MG_DL_TO_MMOL_L - 3.5
    hdl = (i["hdl_mg_dl"] * MG_DL_TO_MMOL_L - 1.3) / 0.3
    sbp_high = (max(i["systolic_bp"], 110) - 130) / 20
    egfr_low = (min(i["egfr"], 60) - 60) / -15
    diabetes, smoker, treated, statin = (
        float(i[name]) for name in ("diabetes", "current_smoker", "antihypertensive", "statin")
    )
    terms = (
        age,
        non_hdl,
        hdl,
        (min(i["systolic_bp"], 110) - 110) / 20,
        sbp_high,
        diabetes,
        smoker,
        egfr_low,
        (max(i["egfr"], 60) - 90) / -15,
        treated,
        statin,
        treated * sbp_high,
        statin * non_hdl,
        age * non_hdl,
        age * hdl,
        age * sbp_high,
        age * diabetes,
        age * smoker,
        age * egfr_low,
        1.0,
    )
    log_odds = sum(
        beta * term for beta, term in zip(PREVENT_TOTAL_CVD[i["sex"]], terms, strict=True)
    )
    return 100 / (1 + math.exp(-log_odds))


_register(
    ClinicalScore(
        id="prevent_cvd_10y",
        name="PREVENT 10-year total cardiovascular disease risk",
        description=(
            "Sex-specific, race-free base equation for 10-year risk of total cardiovascular "
            "disease, using eGFR from the CKD-EPI 2021 equation."
        ),
        citation=(
            "Khan SS, et al. Development and validation of the American Heart Association's "
            "PREVENT equations. Circulation. 2024;149:430-449. "
            "doi:10.1161/CIRCULATIONAHA.123.067626. Coefficients and input ranges: "
            "Supplemental Table S12A, simplified base model."
        ),
        fields=(
            ScoreField(
                "age", "integer", "Age in completed years", unit="a", minimum=30, maximum=79
            ),
            SEX,
            ScoreField(
                "total_cholesterol_mg_dl",
                "number",
                "Total cholesterol",
                unit="mg/dL",
                minimum=130,
                maximum=320,
                conversions=CHOLESTEROL_CONVERSIONS,
            ),
            ScoreField(
                "hdl_mg_dl",
                "number",
                "HDL cholesterol",
                unit="mg/dL",
                minimum=20,
                maximum=100,
                conversions=CHOLESTEROL_CONVERSIONS,
            ),
            ScoreField(
                "systolic_bp",
                "number",
                "Systolic blood pressure",
                unit="mm[Hg]",
                minimum=90,
                maximum=200,
            ),
            RISK_DIABETES,
            CURRENT_SMOKER,
            ScoreField(
                "egfr",
                "number",
                "eGFR from the CKD-EPI 2021 creatinine equation",
                unit="mL/min/{1.73_m2}",
                minimum=15,
                maximum=150,
            ),
            ANTIHYPERTENSIVE,
            _flag("statin", "Current statin"),
        ),
        compute=_prevent_cvd_10y,
        units="%",
        code_system=SCORE_SYSTEM,
        code="prevent_cvd_10y",
        display="PREVENT 10-year total CVD risk",
        tolerance=0.1,
        precision=1,
    )
)

PCE: dict[tuple[str, str], dict[str, float]] = {
    ("female", "white"): {
        "ln_age": -29.799,
        "ln_age_squared": 4.884,
        "ln_tc": 13.540,
        "ln_age_ln_tc": -3.114,
        "ln_hdl": -13.578,
        "ln_age_ln_hdl": 3.149,
        "ln_treated_sbp": 2.019,
        "ln_untreated_sbp": 1.957,
        "smoker": 7.574,
        "ln_age_smoker": -1.665,
        "diabetes": 0.661,
        "mean": -29.18,
        "baseline": 0.9665,
    },
    ("female", "african_american"): {
        "ln_age": 17.114,
        "ln_tc": 0.940,
        "ln_hdl": -18.920,
        "ln_age_ln_hdl": 4.475,
        "ln_treated_sbp": 29.291,
        "ln_age_ln_treated_sbp": -6.432,
        "ln_untreated_sbp": 27.820,
        "ln_age_ln_untreated_sbp": -6.087,
        "smoker": 0.691,
        "diabetes": 0.874,
        "mean": 86.61,
        "baseline": 0.9533,
    },
    ("male", "white"): {
        "ln_age": 12.344,
        "ln_tc": 11.853,
        "ln_age_ln_tc": -2.664,
        "ln_hdl": -7.990,
        "ln_age_ln_hdl": 1.769,
        "ln_treated_sbp": 1.797,
        "ln_untreated_sbp": 1.764,
        "smoker": 7.837,
        "ln_age_smoker": -1.795,
        "diabetes": 0.658,
        "mean": 61.18,
        "baseline": 0.9144,
    },
    ("male", "african_american"): {
        "ln_age": 2.469,
        "ln_tc": 0.302,
        "ln_hdl": -0.307,
        "ln_treated_sbp": 1.916,
        "ln_untreated_sbp": 1.809,
        "smoker": 0.549,
        "diabetes": 0.645,
        "mean": 19.54,
        "baseline": 0.8954,
    },
}


def _pooled_cohort(i: Mapping[str, Any]) -> float:
    coefficients = PCE[(i["sex"], i["race"])]
    ln_age = math.log(i["age"])
    ln_tc = math.log(i["total_cholesterol_mg_dl"])
    ln_hdl = math.log(i["hdl_mg_dl"])
    ln_sbp = math.log(i["systolic_bp"])
    treatment = "treated" if i["antihypertensive"] else "untreated"
    smoker = float(i["current_smoker"])
    terms = {
        "ln_age": ln_age,
        "ln_age_squared": ln_age**2,
        "ln_tc": ln_tc,
        "ln_age_ln_tc": ln_age * ln_tc,
        "ln_hdl": ln_hdl,
        "ln_age_ln_hdl": ln_age * ln_hdl,
        f"ln_{treatment}_sbp": ln_sbp,
        f"ln_age_ln_{treatment}_sbp": ln_age * ln_sbp,
        "smoker": smoker,
        "ln_age_smoker": ln_age * smoker,
        "diabetes": float(i["diabetes"]),
    }
    total = sum(coefficients.get(name, 0.0) * value for name, value in terms.items())
    return float(100 * (1 - coefficients["baseline"] ** math.exp(total - coefficients["mean"])))


def _ascvd_band(value: float, _: Mapping[str, Any]) -> str:
    if value < 5:
        return "low"
    if value < 7.5:
        return "borderline"
    if value < 20:
        return "intermediate"
    return "high"


_register(
    ClinicalScore(
        id="ascvd_pce_10y",
        name="Pooled Cohort Equations 10-year ASCVD risk",
        description=(
            "Sex- and race-specific 10-year risk of a first hard atherosclerotic "
            "cardiovascular event, with the 2019 primary prevention risk categories."
        ),
        citation=(
            "Goff DC Jr, et al. 2013 ACC/AHA guideline on the assessment of cardiovascular "
            "risk. Circulation. 2014;129(25 Suppl 2):S49-S73. "
            "doi:10.1161/01.cir.0000437741.48606.98. Coefficients: Table A. Categories: "
            "Arnett DK, et al. 2019 ACC/AHA guideline on the primary prevention of "
            "cardiovascular disease. Circulation. 2019;140:e596-e646. "
            "doi:10.1161/CIR.0000000000000678."
        ),
        fields=(
            ScoreField(
                "age", "integer", "Age in completed years", unit="a", minimum=40, maximum=79
            ),
            SEX,
            ScoreField(
                "race",
                "enum",
                "Race group that selects the published equation",
                allowed=("white", "african_american"),
            ),
            TOTAL_CHOLESTEROL,
            HDL_CHOLESTEROL,
            SYSTOLIC,
            ANTIHYPERTENSIVE,
            RISK_DIABETES,
            CURRENT_SMOKER,
        ),
        compute=_pooled_cohort,
        units="%",
        code_system=LOINC_SYSTEM,
        code="79423-0",
        display="Cardiovascular disease 10Y risk [Likelihood] ACC-AHA Pooled Cohort by Goff 2013",
        tolerance=0.1,
        precision=1,
        band=_ascvd_band,
    )
)


@dataclass(frozen=True)
class _FraminghamModel:
    age: float
    age_squared: float
    cholesterol: tuple[float, ...]
    hdl: tuple[float, ...]
    blood_pressure: tuple[float, ...]
    diabetes: float
    smoker: float
    mean: float
    baseline: float


FRAMINGHAM_CHD = {
    "male": _FraminghamModel(
        age=0.04826,
        age_squared=0,
        cholesterol=(-0.65945, 0, 0.17692, 0.50539, 0.65713),
        hdl=(0.49744, 0.24310, 0, -0.05107, -0.48660),
        blood_pressure=(-0.00226, 0, 0.28320, 0.52168, 0.61859),
        diabetes=0.42839,
        smoker=0.52337,
        mean=3.0975,
        baseline=0.90015,
    ),
    "female": _FraminghamModel(
        age=0.33766,
        age_squared=-0.00268,
        cholesterol=(-0.26138, 0, 0.20771, 0.24385, 0.53513),
        hdl=(0.84312, 0.37796, 0.19785, 0, -0.42951),
        blood_pressure=(-0.53363, 0, -0.06773, 0.26288, 0.46573),
        diabetes=0.59626,
        smoker=0.29246,
        mean=9.92545,
        baseline=0.96246,
    ),
}


def _framingham_chd(i: Mapping[str, Any]) -> float:
    model = FRAMINGHAM_CHD[i["sex"]]
    # JNC-V stages: when systolic and diastolic fall in different stages, the higher applies.
    stage = max(
        bisect.bisect_right((120, 130, 140, 160), i["systolic_bp"]),
        bisect.bisect_right((80, 85, 90, 100), i["diastolic_bp"]),
    )
    linear = (
        model.age * i["age"]
        + model.age_squared * i["age"] ** 2
        + model.cholesterol[bisect.bisect_right((160, 200, 240, 280), i["total_cholesterol_mg_dl"])]
        + model.hdl[bisect.bisect_right((35, 45, 50, 60), i["hdl_mg_dl"])]
        + model.blood_pressure[stage]
        + model.diabetes * float(i["diabetes"])
        + model.smoker * float(i["current_smoker"])
    )
    return float(100 * (1 - model.baseline ** math.exp(linear - model.mean)))


_register(
    ClinicalScore(
        id="framingham_chd_10y",
        name="Framingham 10-year coronary heart disease risk",
        description=(
            "10-year coronary heart disease risk from age, total and HDL cholesterol "
            "categories, JNC-V blood pressure stage, diabetes and smoking."
        ),
        citation=(
            "Wilson PWF, et al. Prediction of coronary heart disease using risk factor "
            "categories. Circulation. 1998;97:1837-1847. doi:10.1161/01.CIR.97.18.1837. "
            "Coefficients: Table 6, total cholesterol categories."
        ),
        fields=(
            ScoreField(
                "age", "integer", "Age in completed years", unit="a", minimum=30, maximum=74
            ),
            SEX,
            TOTAL_CHOLESTEROL,
            HDL_CHOLESTEROL,
            SYSTOLIC,
            DIASTOLIC,
            RISK_DIABETES,
            CURRENT_SMOKER,
        ),
        compute=_framingham_chd,
        units="%",
        code_system=LOINC_SYSTEM,
        code="65851-8",
        display="Coronary heart disease 10Y risk [#] Framingham.Wilson 1998",
        tolerance=0.1,
        precision=1,
    )
)


def score_spec(score_id: str) -> dict[str, Any]:
    """Return the fields and metadata of one score."""
    return _score(score_id).spec()


def _score(score_id: str) -> ClinicalScore:
    if score_id not in SCORES:
        raise ScoreInputError("unknown_score", f"unknown score: {score_id}", invalid=["score"])
    return SCORES[score_id]


def _valid(field: ScoreField, value: Any) -> bool:
    if field.type == "boolean":
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    if field.type == "enum":
        return value in (field.allowed or ())
    if field.type == "integer" and not isinstance(value, int):
        return False
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    if field.allowed is not None and value not in field.allowed:
        return False
    if field.minimum is not None and value < field.minimum:
        return False
    if field.minimum == 0 and field.type == "number" and value == 0:
        return False
    return field.maximum is None or value <= field.maximum


def calculate(score_id: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Validate every input and compute value, band and units; never default a field."""
    score = _score(score_id)
    names = {field.name for field in score.fields}
    missing = [field.name for field in score.fields if inputs.get(field.name) is None]
    invalid = sorted(name for name in inputs if name not in names)
    invalid += [
        field.name
        for field in score.fields
        if inputs.get(field.name) is not None and not _valid(field, inputs[field.name])
    ]
    if missing or invalid:
        raise ScoreInputError(
            "invalid_inputs",
            f"{score_id} needs valid values for every field",
            missing=missing,
            invalid=invalid,
        )
    raw = score.compute(inputs)
    value: float | int = round(raw) if score.integer else round(raw, score.precision)
    return {
        "score": score.id,
        "value": value,
        "band": score.band(raw, inputs) if score.band else None,
        "units": score.units,
    }
