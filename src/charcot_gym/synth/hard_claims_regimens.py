"""Calculate drug regimens and diagnosis links."""

from __future__ import annotations

import random
from decimal import ROUND_HALF_UP, Decimal
from typing import Any


TRASTUZUMAB_PRODUCTS = {
    "J9355": {
        "brand": "Herceptin",
        "generic": "trastuzumab",
        "ndc_150": "50242-132-01",
        "ndc_420": None,
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=492dbdb2-077e-4064-bff3-372d6af0a7a2",
    },
    "Q5112": {
        "brand": "Ontruzant",
        "generic": "trastuzumab-dttb",
        "ndc_150": "78206-147-01",
        "ndc_420": "78206-148-01",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ce2a2492-85e3-4b0a-872a-047f44e4203d",
    },
    "Q5113": {
        "brand": "Herzuma",
        "generic": "trastuzumab-pkrb",
        "ndc_150": "63459-303-43",
        "ndc_420": "63459-305-47",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=ae71c003-883a-4cd8-ad50-e8cc9a54e971",
    },
    "Q5114": {
        "brand": "Ogivri",
        "generic": "trastuzumab-dkst",
        "ndc_150": "83257-001-11",
        "ndc_420": "83257-003-01",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=b6465b44-a6dd-f99c-d009-6851cf05169c",
    },
    "Q5116": {
        "brand": "Trazimera",
        "generic": "trastuzumab-qyyp",
        "ndc_150": "0069-0308-01",
        "ndc_420": "0069-0305-01",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=b9c5e894-27d2-4245-a653-df986fed3c56",
    },
    "Q5117": {
        "brand": "Kanjinti",
        "generic": "trastuzumab-anns",
        "ndc_150": "55513-141-01",
        "ndc_420": "55513-132-01",
        "source_url": "https://dailymed.nlm.nih.gov/dailymed/drugInfo.cfm?setid=9a94f5b2-b84e-4987-8ea1-c4346db5a5fa",
    },
}


def _regimen(  # noqa: PLR0913
    regimen: str,
    index: int,
    docetaxel_dose: int,
    bsa: float,
    *,
    rng: random.Random,
    include_cyclophosphamide: bool = False,
    trastuzumab_product: dict[str, Any] | None = None,
    trastuzumab_dose_mg: int | None = None,
    trastuzumab_cycle: int | None = None,
    trastuzumab_presentation: int | None = None,
) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    breast = "C50.911" if index % 2 else "C50.912"
    cancer_code = "C61" if regimen == "metastatic_prostate" else breast
    diagnoses = ["Z51.11", cancer_code]
    diagnosis_by_code = {"J9171": 2, "J2405": 1, "J9075": 2}
    docetaxel_container_type = "single_dose" if rng.random() < 2 / 3 else "multi_dose"
    docetaxel_container_sizes = (
        [20, 80, 160] if docetaxel_container_type == "single_dose" else [80, 160]
    )
    docetaxel_container_labels = {
        20: "20 mg/1 mL single-dose vial",
        80: f"80 mg/4 mL {docetaxel_container_type.replace('_', '-')} vial",
        160: f"160 mg/8 mL {docetaxel_container_type.replace('_', '-')} vial",
    }
    administrations: list[dict[str, Any]] = [
        {
            "code": "J9171",
            "name": "Docetaxel",
            "dose": docetaxel_dose,
            "dose_unit": "mg",
            "diagnosis_sequence": 2,
            "container_type": docetaxel_container_type,
            "container_sizes_mg": docetaxel_container_sizes,
            "container_labels": docetaxel_container_labels,
        }
    ]
    if (
        regimen == "early_breast"
        and trastuzumab_product is None
        and (index % 2 == 0 or include_cyclophosphamide)
    ):
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
                    500: "500 mg single-dose vial (powder)",
                    1000: "1 g single-dose vial (powder)",
                    2000: "2 g single-dose vial (powder)",
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
    if trastuzumab_product is not None and trastuzumab_dose_mg is not None:
        code = trastuzumab_product["code"]
        presentation = trastuzumab_presentation or 150
        ndc = trastuzumab_product["ndc_150" if presentation == 150 else "ndc_420"]
        administrations.append(
            {
                "code": code,
                "name": f"{trastuzumab_product['brand']} ({trastuzumab_product['generic']})",
                "dose": trastuzumab_dose_mg,
                "dose_unit": "mg",
                "diagnosis_sequence": 2,
                "container_type": "single_dose" if presentation == 150 else "multi_dose",
                "container_sizes_mg": [presentation],
                "container_labels": {
                    presentation: (
                        f"{presentation} mg "
                        f"{'single-dose' if presentation == 150 else 'multi-dose'} vial"
                    )
                },
                "ndc_by_size": {presentation: ndc},
                "cycle": trastuzumab_cycle,
            }
        )
        diagnosis_by_code[code] = 2
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
            "ndc_by_size": {40: "16729-298-05"},
        }
    )
    return administrations, diagnoses, diagnosis_by_code
