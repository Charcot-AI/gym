"""Calculate drug regimens and diagnosis links."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from typing import Any


def _regimen(
    regimen: str,
    index: int,
    docetaxel_dose: int,
    bsa: float,
    include_cyclophosphamide: bool = False,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    breast = "C50.911" if index % 2 else "C50.912"
    cancer_code = "C61" if regimen == "metastatic_prostate" else breast
    diagnoses = ["Z51.11", cancer_code]
    diagnosis_by_code = {"J9171": 2, "J2405": 1, "J9075": 2}
    administrations: list[dict[str, Any]] = [
        {
            "code": "J9171",
            "name": "Docetaxel",
            "dose": docetaxel_dose,
            "dose_unit": "mg",
            "diagnosis_sequence": 2,
            "container_type": "multi_dose",
            "container_sizes_mg": [80, 160],
            "container_labels": {
                80: "80 mg/4 mL multi-dose vial",
                160: "160 mg/8 mL multi-dose vial",
            },
        }
    ]
    if regimen == "early_breast" and (index % 2 == 0 or include_cyclophosphamide):
        cyclophosphamide_dose = int(
            Decimal(str(600 * bsa / 5)).quantize(Decimal("1"), rounding=ROUND_HALF_UP) * 5
        )
        administrations.append(
            {
                "code": "J9075",
                "name": "Cyclophosphamide",
                "dose": cyclophosphamide_dose,
                "dose_unit": "mg",
                "diagnosis_sequence": 2,
                "container_type": "single_dose",
                "container_sizes_mg": [500, 1000, 2000],
                "container_labels": {
                    500: "500 mg/mL single-dose vial",
                    1000: "1 g/2 mL single-dose vial",
                    2000: "2 g/4 mL single-dose vial",
                },
            }
        )
    if regimen == "metastatic_breast_bone":
        diagnoses.append("C79.51")
        diagnosis_by_code["J0897"] = 3
        administrations.append(
            {
                "code": "J0897",
                "name": "Denosumab (Xgeva)",
                "dose": 120,
                "dose_unit": "mg",
                "diagnosis_sequence": 3,
                "vials_mg": [120],
                "waste_mg": 0,
                "container_type": "single_dose",
                "container_sizes_mg": [120],
                "container_labels": {120: "120 mg/1.7 mL single-dose vial"},
            }
        )
    administrations.append(
        {
            "code": "J2405",
            "name": "Ondansetron",
            "dose": 8,
            "dose_unit": "mg",
            "diagnosis_sequence": 1,
            "container_type": "multi_dose",
            "container_sizes_mg": [40],
            "container_labels": {40: "40 mg/20 mL multi-dose vial"},
        }
    )
    return administrations, diagnoses, diagnosis_by_code
