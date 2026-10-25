"""Search the bundled terminology subsets."""

from __future__ import annotations

import json
import re
from datetime import date
from functools import lru_cache
from importlib.resources import files
from typing import Any

LOINC_SYSTEM_URI = "http://loinc.org"
UCUM_SYSTEM_URI = "http://unitsofmeasure.org"
ICD10CM_SYSTEM_URI = "http://hl7.org/fhir/sid/icd-10-cm"
HCPCS_SYSTEM_URI = "http://www.cms.gov/Medicare/Coding/HCPCSReleaseCodeSets"
HL7_HCPCS_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/HCPCS"
CONDITION_CLINICAL_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/condition-clinical"
CONDITION_VERIFICATION_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/condition-ver-status"
CONDITION_CATEGORY_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/condition-category"
CLAIM_TYPE_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/claim-type"
RELATED_CLAIM_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/ex-relatedclaimrelationship"
OBSERVATION_CATEGORY_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/observation-category"
V3_ACT_CODE_SYSTEM_URI = "http://terminology.hl7.org/CodeSystem/v3-ActCode"
NDC_SYSTEM_URI = "http://hl7.org/fhir/sid/ndc"

CLAIM_STATUS_ACTIVE = "active"
CLAIM_TYPE_PROFESSIONAL = "professional"
CLAIM_USE_VALUES = ("claim", "preauthorization")
RELATED_CLAIM_PRIOR = "prior"
DOCUMENT_STATUS_FINAL = "final"
CONDITION_CLINICAL_ACTIVE = "active"
CONDITION_VERIFICATION_CONFIRMED = "confirmed"
CONDITION_CATEGORY_ENCOUNTER_DIAGNOSIS = "encounter-diagnosis"
INSURANCE_FOCAL = "focal"
ENCOUNTER_CLASS_AMBULATORY = "AMB"

TOKEN_PATTERN = re.compile(r"[\w*]+(?:[-/][\w*]+)*", re.UNICODE)


def terminology_search(system: str, query: str, limit: int = 10) -> list[dict[str, Any]]:
    """Prefer exact synonym matches, then search display and synonym tokens."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    entries = _entries()
    exact_matches = [
        entry
        for entry in entries
        if entry["system"] == system
        and any(synonym.casefold() == query.strip().casefold() for synonym in entry["synonyms"])
    ]
    search_entries = exact_matches or entries
    matches = []
    for entry in search_entries:
        if entry["system"] != system:
            continue
        searchable = " ".join([entry["display"], *entry["synonyms"]])
        if query_tokens <= _tokens(searchable):
            today = date.today()
            add_date = date.fromisoformat(entry["add_date"]) if entry.get("add_date") else None
            termination_date = (
                date.fromisoformat(entry["termination_date"])
                if entry.get("termination_date")
                else None
            )
            status = (
                "active"
                if (add_date is None or add_date <= today)
                and (termination_date is None or today <= termination_date)
                else "inactive"
            )
            matches.append(
                {
                    "system": entry["system"],
                    "code": entry["code"],
                    "display": entry["display"],
                    "unit": entry.get("unit", ""),
                    "status": status,
                    "termination_date": termination_date.isoformat() if termination_date else None,
                }
            )
    return matches[:limit]


def hcpcs_is_active_on(code: str, service_date: str) -> bool:
    """Return whether a bundled HCPCS code is active on a service date."""
    try:
        billed_on = date.fromisoformat(service_date)
    except (TypeError, ValueError):
        return False
    for entry in _entries():
        if entry["system"] != HCPCS_SYSTEM_URI:
            continue
        if entry["code"] != code:
            continue
        add_date = date.fromisoformat(entry["add_date"]) if entry.get("add_date") else None
        termination_date = (
            date.fromisoformat(entry["termination_date"]) if entry.get("termination_date") else None
        )
        return (add_date is None or add_date <= billed_on) and (
            termination_date is None or billed_on <= termination_date
        )
    return False


@lru_cache(maxsize=1)
def _entries() -> tuple[dict[str, Any], ...]:
    """Read the bundled subset once per process."""
    data = files("charcot_gym").joinpath("data")
    entries = []
    for filename in (
        "loinc_subset.json",
        "ucum_subset.json",
        "icd10cm_subset.json",
        "hcpcs_subset.json",
    ):
        content = data.joinpath(filename).read_text(encoding="utf-8")
        subset = json.loads(content)
        if not isinstance(subset, list):
            raise ValueError("terminology data must be a list")
        entries.extend(subset)
    return tuple(entries)


def _tokens(value: str) -> set[str]:
    tokens = set()
    for token in TOKEN_PATTERN.findall(value.casefold()):
        tokens.add(token)
        tokens.update(part for part in re.split(r"[-/]+", token) if part)
    return tokens
