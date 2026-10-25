"""Build FHIR resources for Hard Claims tasks."""

# Resource builders keep FHIR fields explicit for direct inspection.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import base64
import copy
from datetime import date
from decimal import ROUND_CEILING, Decimal
from typing import Any

from charcot_gym.terminology import (
    CLAIM_TYPE_PROFESSIONAL,
    CLAIM_TYPE_SYSTEM_URI,
    CONDITION_CLINICAL_SYSTEM_URI,
    HCPCS_SYSTEM_URI,
    ICD10CM_SYSTEM_URI,
    LOINC_SYSTEM_URI,
    NDC_SYSTEM_URI,
    OBSERVATION_CATEGORY_SYSTEM_URI,
    RELATED_CLAIM_PRIOR,
    RELATED_CLAIM_SYSTEM_URI,
    UCUM_SYSTEM_URI,
    V3_ACT_CODE_SYSTEM_URI,
)

CLAIM_TYPE = {"coding": [{"system": CLAIM_TYPE_SYSTEM_URI, "code": CLAIM_TYPE_PROFESSIONAL}]}
RELATED_PRIOR = {
    "coding": [
        {
            "system": RELATED_CLAIM_SYSTEM_URI,
            "code": RELATED_CLAIM_PRIOR,
        }
    ]
}


def _claim(
    prior_items: list[tuple[dict[str, Any], date]],
    diagnoses: list[str],
    diagnosis_by_code: dict[str, int],
    hcpcs: dict[str, dict[str, Any]],
    plan: dict[str, Any],
    patient_id: str,
    provider_id: str,
    insurance: list[dict[str, Any]],
    service_date: date,
    fee_rows: dict[str, list[dict[str, Any]]],
    packs: list[str],
    cms_modifiers: bool,
    preauth_ref: str | None = None,
    use: str = "claim",
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    total = Decimal("0")
    for administration, item_date in prior_items:
        code = administration["code"]
        unit_size = Decimal(str(hcpcs[code]["unit_size"]))
        dose = Decimal(str(administration["dose"]))
        quantity = int((dose / unit_size).to_integral_value(rounding=ROUND_CEILING))
        price = _price(code, item_date, plan, fee_rows)
        item = _claim_item(
            code,
            quantity,
            item_date,
            price,
            diagnosis_by_code[code],
            hcpcs,
            len(items) + 1,
            ndc=administration.get("ndc_by_size", {}).get(administration.get("vials_mg", [None])[0])
            if plan.get("ndc_reporting_required")
            else None,
        )
        single_dose = administration.get("container_type") == "single_dose"
        waste = int(administration.get("waste_mg", 0)) if cms_modifiers and single_dose else 0
        if cms_modifiers and single_dose and waste:
            container_quantity = int(
                (
                    sum(Decimal(str(size)) for size in administration["vials_mg"]) / unit_size
                ).to_integral_value(rounding=ROUND_CEILING)
            )
            waste_quantity = container_quantity - quantity
            jw_item = _claim_item(
                code,
                waste_quantity,
                item_date,
                price,
                diagnosis_by_code[code],
                hcpcs,
                len(items) + 2,
                "JW",
                ndc=administration.get("ndc_by_size", {}).get(
                    administration.get("vials_mg", [None])[0]
                )
                if plan.get("ndc_reporting_required")
                else None,
            )
            items.extend([item, jw_item])
            total += Decimal(str(item["net"]["value"])) + Decimal(str(jw_item["net"]["value"]))
        else:
            if cms_modifiers and single_dose:
                item["modifier"] = [{"coding": [{"system": HCPCS_SYSTEM_URI, "code": "JZ"}]}]
            items.append(item)
            total += Decimal(str(item["net"]["value"]))
    claim: dict[str, Any] = {
        "resourceType": "Claim",
        "status": "active",
        "use": use,
        "type": CLAIM_TYPE,
        "patient": {"reference": f"Patient/{patient_id}"},
        "created": service_date.isoformat(),
        "insurer": {"reference": f"Organization/{plan['healthplan_id']}"},
        "provider": {"reference": f"Organization/{provider_id}"},
        "insurance": copy.deepcopy(insurance),
        "diagnosis": [
            {
                "sequence": sequence,
                "diagnosisCodeableConcept": {
                    "coding": [{"system": ICD10CM_SYSTEM_URI, "code": code}]
                },
            }
            for sequence, code in enumerate(diagnoses, 1)
        ],
        "item": items,
        "total": {"value": float(total), "currency": "USD"},
    }
    if preauth_ref and use == "claim":
        claim["insurance"][0]["preAuthRef"] = [preauth_ref]
    return claim


def _claim_item(
    code: str,
    quantity: int,
    item_date: date,
    price: Decimal,
    diagnosis_sequence: int,
    hcpcs: dict[str, dict[str, Any]],
    sequence: int,
    modifier: str | None = None,
    *,
    ndc: str | None = None,
) -> dict[str, Any]:
    item = {
        "sequence": sequence,
        "productOrService": {"coding": [{"system": hcpcs[code]["system"], "code": code}]},
        "quantity": {"value": quantity},
        "servicedDate": item_date.isoformat(),
        "unitPrice": {"value": float(price), "currency": "USD"},
        "net": {"value": float(price * quantity), "currency": "USD"},
        "diagnosisSequence": [diagnosis_sequence],
    }
    if modifier:
        item["modifier"] = [{"coding": [{"system": HCPCS_SYSTEM_URI, "code": modifier}]}]
    if ndc:
        item["detail"] = [
            {
                "sequence": 1,
                "productOrService": {
                    "coding": [{"system": NDC_SYSTEM_URI, "code": _normalize_ndc(ndc)}]
                },
            }
        ]
    return item


def _normalize_ndc(value: str) -> str:
    segments = value.split("-")
    if len(segments) != 3:
        raise ValueError(f"invalid NDC: {value}")
    widths = [len(segment) for segment in segments]
    if widths == [5, 3, 2]:
        segments[1] = segments[1].zfill(4)
    elif widths == [4, 4, 2]:
        segments[0] = segments[0].zfill(5)
    elif widths == [5, 4, 1]:
        segments[2] = segments[2].zfill(2)
    elif widths != [5, 4, 2]:
        raise ValueError(f"invalid NDC: {value}")
    return "".join(segments)


def _price(
    code: str,
    service_date: date,
    plan: dict[str, Any],
    fee_rows: dict[str, list[dict[str, Any]]],
) -> Decimal:
    rows = fee_rows.get(code)
    if rows:
        for row in rows:
            if service_date.isoformat() >= row["start"] and (
                row["end"] is None or service_date.isoformat() <= row["end"]
            ):
                return Decimal(row["price"])
    return Decimal(str(plan["fee_schedule"][code]))


def _condition(
    resource_id: str,
    patient_id: str,
    encounter_id: str,
    code: str,
    recorded_date: date,
) -> dict[str, Any]:
    return {
        "resourceType": "Condition",
        "id": resource_id,
        "clinicalStatus": {
            "coding": [
                {
                    "system": CONDITION_CLINICAL_SYSTEM_URI,
                    "code": "active",
                }
            ]
        },
        "code": {"coding": [{"system": ICD10CM_SYSTEM_URI, "code": code}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "recordedDate": recorded_date.isoformat(),
    }


def _observation(
    resource_id: str,
    patient_id: str,
    observed: date,
    code: str,
    value: float,
    unit: str,
) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": resource_id,
        "status": "final",
        "category": [
            {
                "coding": [
                    {
                        "system": OBSERVATION_CATEGORY_SYSTEM_URI,
                        "code": "vital-signs",
                    }
                ]
            }
        ],
        "code": {"coding": [{"system": LOINC_SYSTEM_URI, "code": code}]},
        "subject": {"reference": f"Patient/{patient_id}"},
        "effectiveDateTime": observed.isoformat(),
        "valueQuantity": {"value": value, "unit": unit, "system": UCUM_SYSTEM_URI, "code": unit},
    }


def _coverage_resource(resource_id: str, patient_id: str, plan_id: str) -> dict[str, Any]:
    return {
        "resourceType": "Coverage",
        "id": resource_id,
        "status": "active",
        "beneficiary": {"reference": f"Patient/{patient_id}"},
        "payor": [{"reference": f"Organization/{plan_id}"}],
        "period": {"start": "2020-01-01", "end": "2027-12-31"},
    }


def _encounter(resource_id: str, patient_id: str, service_date: date) -> dict[str, Any]:
    return {
        "resourceType": "Encounter",
        "id": resource_id,
        "status": "completed",
        "class": [{"coding": [{"system": V3_ACT_CODE_SYSTEM_URI, "code": "AMB"}]}],
        "subject": {"reference": f"Patient/{patient_id}"},
        "actualPeriod": {"start": service_date.isoformat()},
    }


def _document(
    resource_id: str,
    patient_id: str,
    encounter_id: str,
    service_date: date,
    text: str,
    title: str,
) -> dict[str, Any]:
    return {
        "resourceType": "DocumentReference",
        "id": resource_id,
        "status": "current",
        "description": title,
        "subject": {"reference": f"Patient/{patient_id}"},
        "date": service_date.isoformat(),
        "context": {"encounter": [{"reference": f"Encounter/{encounter_id}"}]},
        "content": [
            {
                "attachment": {
                    "contentType": "text/plain",
                    "data": base64.b64encode(text.encode()).decode(),
                }
            }
        ],
    }


def _organization(resource_id: str, name: str) -> dict[str, str]:
    return {"resourceType": "Organization", "id": resource_id, "name": name}


def _medication_administration(
    resource_id: str,
    patient_id: str,
    encounter_id: str,
    service_date: date,
    item: dict[str, Any],
) -> dict[str, Any]:
    return {
        "resourceType": "MedicationAdministration",
        "id": resource_id,
        "status": "completed",
        "medication": {"concept": {"text": item["name"]}},
        "subject": {"reference": f"Patient/{patient_id}"},
        "encounter": {"reference": f"Encounter/{encounter_id}"},
        "effectiveDateTime": service_date.isoformat(),
        "dosage": {"dose": {"value": float(item["dose"]), "unit": item["dose_unit"]}},
    }


def _preauth(
    resource_id: str, patient_id: str, plan_id: str, created: date, reference: str
) -> dict[str, Any]:
    return {
        "resourceType": "ClaimResponse",
        "id": resource_id,
        "status": "active",
        "type": CLAIM_TYPE,
        "use": "preauthorization",
        "patient": {"reference": f"Patient/{patient_id}"},
        "created": created.isoformat(),
        "insurer": {"reference": f"Organization/{plan_id}"},
        "outcome": "complete",
        "preAuthRef": reference,
    }


def _denial(
    resource_id: str, patient_id: str, plan_id: str, created: date, request_id: str
) -> dict[str, Any]:
    return {
        "resourceType": "ClaimResponse",
        "id": resource_id,
        "status": "active",
        "type": CLAIM_TYPE,
        "use": "claim",
        "patient": {"reference": f"Patient/{patient_id}"},
        "created": created.isoformat(),
        "insurer": {"reference": f"Organization/{plan_id}"},
        "request": {"reference": f"Claim/{request_id}"},
        "outcome": "error",
        "error": [
            {
                "itemSequence": 1,
                "code": {
                    "coding": [
                        {
                            "system": "https://charcot.ai/fhir/CodeSystem/denial-reason",
                            "code": "administered-dose-units",
                        }
                    ]
                },
            }
        ],
    }


def _has_modifier(item: dict[str, Any], code: str) -> bool:
    return any(
        coding.get("code") == code
        for modifier in item.get("modifier", [])
        for coding in modifier.get("coding", [])
    )


def _product_code(item: dict[str, Any]) -> str | None:
    return next(
        (
            coding.get("code")
            for coding in item.get("productOrService", {}).get("coding", [])
            if isinstance(coding, dict)
        ),
        None,
    )


def _reprice(claim: dict[str, Any]) -> None:
    total = Decimal("0")
    for item in claim["item"]:
        value = Decimal(str(item["unitPrice"]["value"])) * Decimal(str(item["quantity"]["value"]))
        item["net"]["value"] = float(value)
        total += value
    claim["total"]["value"] = float(total)
