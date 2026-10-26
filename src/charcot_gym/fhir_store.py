"""Deterministic in-memory storage for the supported FHIR R5 resource shapes."""

from __future__ import annotations

import re
from copy import deepcopy
from datetime import UTC, date, datetime, time, timedelta
from typing import Any

SUPPORTED_TYPES = {
    "Patient",
    "Observation",
    "Condition",
    "Encounter",
    "DocumentReference",
    "Organization",
    "Coverage",
    "MedicationAdministration",
    "MedicationStatement",
    "QuestionnaireResponse",
    "ClaimResponse",
    "Claim",
    "Task",
}
REFERENCE_PATTERN = re.compile(r"^([A-Z][A-Za-z]+)/([A-Za-z0-9.-]{1,64})$")
REQUIRED_FIELDS = {
    "Patient": (),
    "Observation": ("status", "code", "subject"),
    "Condition": ("subject", "code"),
    "Encounter": ("status", "class"),
    "DocumentReference": ("status", "content"),
    "Organization": (),
    "Coverage": ("status", "beneficiary", "payor"),
    "MedicationAdministration": ("status", "medication", "subject"),
    "MedicationStatement": ("status", "medication", "subject"),
    "QuestionnaireResponse": ("status", "subject"),
    "ClaimResponse": ("status", "type", "use", "patient", "outcome"),
    "Claim": ("status", "type", "use", "patient"),
    "Task": ("status", "intent"),
}
PATIENT_REFERENCE_FIELDS = {
    "Observation": "subject",
    "Condition": "subject",
    "Encounter": "subject",
    "DocumentReference": "subject",
    "MedicationAdministration": "subject",
    "MedicationStatement": "subject",
    "QuestionnaireResponse": "subject",
    "Coverage": "beneficiary",
    "Claim": "patient",
    "ClaimResponse": "patient",
    "Task": "for",
}
SUBJECT_SEARCH_TYPES = {
    "Observation",
    "Condition",
    "Encounter",
    "DocumentReference",
    "MedicationAdministration",
    "MedicationStatement",
    "QuestionnaireResponse",
}
DATE_FIELDS = {
    "Observation": (("effectiveDateTime",),),
    "Encounter": (("actualPeriod", "start"),),
    "Condition": (("recordedDate",),),
    "DocumentReference": (("date",),),
    "MedicationAdministration": (
        ("occurenceDateTime",),
        ("occurrenceDateTime",),
        ("effectiveDateTime",),
    ),
    "MedicationStatement": (("effectiveDateTime",),),
    "QuestionnaireResponse": (("authored",),),
    "Claim": (("created",),),
}
SUPPORTED_SEARCH_PARAMETERS = {
    "subject",
    "patient",
    "code",
    "date",
    "status",
    "clinical-status",
    "encounter",
    "_sort",
    "_count",
}
DATE_FILTER_PATTERN = re.compile(
    r"^(?:(eq|ge|le|gt|lt))?(\d{4}-\d{2}-\d{2})"
    r"(?:T(\d{2}:\d{2}:\d{2})(Z|[+-]\d{2}:\d{2})?)?$"
)


class FHIRStore:
    """Store isolated resource copies and validate local resource references."""

    def __init__(self, bundle: dict[str, Any], seed: int = 0) -> None:
        if bundle.get("resourceType") != "Bundle":
            raise ValueError("FHIR store must be seeded with a Bundle")
        self._seed = seed
        self._resources: dict[tuple[str, str], dict[str, Any]] = {}
        self._counters: dict[str, int] = {}
        entries = bundle.get("entry", [])
        if not isinstance(entries, list):
            raise ValueError("Bundle.entry must be a list")
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("resource"), dict):
                raise ValueError("Bundle entries must contain a resource")
            resource = deepcopy(entry["resource"])
            resource_type = resource.get("resourceType")
            if resource_type not in SUPPORTED_TYPES:
                raise ValueError(f"unsupported FHIR resource type: {resource_type}")
            resource_id = resource.get("id")
            if not isinstance(resource_id, str) or not resource_id:
                resource_id = self._next_id(str(resource_type))
                resource["id"] = resource_id
            key = (str(resource_type), resource_id)
            if key in self._resources:
                raise ValueError(f"duplicate resource: {resource_type}/{resource_id}")
            self._resources[key] = resource
        for resource in self._resources.values():
            self._validate(resource)

    def snapshot(self) -> dict[str, Any]:
        """Return a detached FHIR collection bundle."""
        return {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": deepcopy(resource)} for resource in self._resources.values()],
        }

    def search(
        self, resource_type: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Search a resource type using supported simple parameters."""
        if resource_type not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported FHIR resource type: {resource_type}")
        filters = {
            name: value for name, value in (params or {}).items() if not _empty_filter(value)
        }
        invalid_params = set(filters) - SUPPORTED_SEARCH_PARAMETERS
        if invalid_params:
            name = sorted(invalid_params)[0]
            supported = ", ".join(sorted(SUPPORTED_SEARCH_PARAMETERS))
            raise ValueError(
                f"OperationOutcome: unsupported search parameter {name!r}. "
                f"Supported parameters: {supported}."
            )
        sort_value = filters.pop("_sort", None)
        if sort_value not in {None, "date", "-date"}:
            raise ValueError("_sort must be either 'date' or '-date'.")
        count = filters.pop("_count", None)
        if count is not None and (
            not isinstance(count, int) or isinstance(count, bool) or count < 0
        ):
            raise ValueError("_count must be a non-negative integer.")

        results = []
        for (kind, _), resource in sorted(self._resources.items()):
            if kind != resource_type:
                continue
            if all(
                self._matches_param(resource_type, resource, name, value)
                for name, value in filters.items()
            ):
                results.append(deepcopy(resource))
        if sort_value is not None:
            results = _sort_by_date(results, resource_type, descending=sort_value == "-date")
        if count is not None:
            results = results[:count]
        return results

    def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        """Return a detached resource matching a type and id."""
        try:
            return deepcopy(self._resources[(resource_type, resource_id)])
        except KeyError as error:
            raise ValueError(f"resource not found: {resource_type}/{resource_id}") from error

    def create(self, resource: dict[str, Any]) -> dict[str, Any]:
        """Validate and store a copy with a deterministic generated id."""
        candidate = deepcopy(resource)
        resource_type = candidate.get("resourceType")
        if resource_type not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported FHIR resource type: {resource_type}")
        candidate["id"] = self._next_id(str(resource_type))
        self._validate(candidate)
        self._resources[(str(resource_type), candidate["id"])] = candidate
        return deepcopy(candidate)

    def update(
        self, resource_type: str, resource_id: str, resource: dict[str, Any]
    ) -> dict[str, Any]:
        """Replace a resource after type, shape, and reference validation."""
        key = (resource_type, resource_id)
        if key not in self._resources:
            raise ValueError(f"resource not found: {resource_type}/{resource_id}")
        candidate = deepcopy(resource)
        if candidate.get("resourceType") != resource_type:
            raise ValueError("resourceType does not match the target")
        candidate["id"] = resource_id
        self._validate(candidate)
        self._resources[key] = candidate
        return deepcopy(candidate)

    def _next_id(self, resource_type: str) -> str:
        counter = self._counters.get(resource_type, 0)
        while True:
            counter += 1
            resource_id = f"{resource_type.lower()}-{self._seed}-{counter}"
            if (resource_type, resource_id) not in self._resources:
                self._counters[resource_type] = counter
                return resource_id

    def _validate(self, resource: dict[str, Any]) -> None:
        resource_type = resource.get("resourceType")
        if resource_type not in SUPPORTED_TYPES:
            raise ValueError(f"unsupported FHIR resource type: {resource_type}")
        for field in REQUIRED_FIELDS[str(resource_type)]:
            if field not in resource:
                raise ValueError(f"{resource_type}.{field} is required")
        if resource_type == "Observation" and not any(
            field in resource
            for field in (
                "valueCodeableConcept",
                "valueQuantity",
                "valueString",
                "valueInteger",
            )
        ):
            raise ValueError("Observation requires a value[x] field")
        if resource_type in {"Observation", "Condition"} and not _has_code(resource.get("code")):
            raise ValueError(f"{resource_type}.code must contain a coding")
        self._validate_references(resource)

    def _validate_references(self, value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("reference")
            if reference is not None:
                if not isinstance(reference, str):
                    raise ValueError("FHIR reference must be a string")
                match = REFERENCE_PATTERN.fullmatch(reference)
                if match is None or (match.group(1), match.group(2)) not in self._resources:
                    raise ValueError(f"reference does not resolve: {reference}")
            for nested in value.values():
                self._validate_references(nested)
        elif isinstance(value, list):
            for nested in value:
                self._validate_references(nested)

    @staticmethod
    def _matches_param(resource_type: str, resource: dict[str, Any], name: str, value: Any) -> bool:
        if name == "subject":
            if resource_type not in SUBJECT_SEARCH_TYPES:
                return True
            return _reference_matches(resource.get("subject"), value)
        if name == "patient":
            field = PATIENT_REFERENCE_FIELDS.get(resource_type)
            return field is not None and _reference_matches(resource.get(field), value)
        if name == "code":
            return _contains_code(resource, value)
        if name == "status":
            return bool(resource.get("status") == value)
        if name == "clinical-status":
            return resource_type == "Condition" and _contains_code(
                resource.get("clinicalStatus"), value
            )
        if name == "encounter":
            return _contains_encounter_reference(resource, value)
        if name == "date":
            date_values = value if isinstance(value, list) else [value]
            return bool(date_values) and all(
                _matches_date(resource_type, resource, item) for item in date_values
            )
        return False


def _empty_filter(value: Any) -> bool:
    return value == "" or (isinstance(value, list) and not [item for item in value if item != ""])


def _reference_matches(value: Any, query: Any, resource_type: str = "Patient") -> bool:
    reference = value.get("reference") if isinstance(value, dict) else None
    if not isinstance(reference, str) or not isinstance(query, str):
        return False
    expected = query if "/" in query else f"{resource_type}/{query}"
    return reference == expected


def _contains_encounter_reference(value: Any, query: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (
                _encounter_reference_matches(item, query)
                if key == "encounter"
                else _contains_encounter_reference(item, query)
            )
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_encounter_reference(item, query) for item in value)
    return False


def _encounter_reference_matches(value: Any, query: Any) -> bool:
    if isinstance(value, list):
        return any(_reference_matches(item, query, "Encounter") for item in value)
    return _reference_matches(value, query, "Encounter")


def _matches_date(resource_type: str, resource: dict[str, Any], query: Any) -> bool:
    if not isinstance(query, str):
        raise ValueError("date filters must be strings.")
    match = DATE_FILTER_PATTERN.fullmatch(query)
    if match is None:
        raise ValueError(
            f"invalid date filter {query!r}; expected [eq|ge|le|gt|lt]YYYY-MM-DD"
            " or a second-precision dateTime."
        )
    prefix = match.group(1) or "eq"
    query_date = date.fromisoformat(match.group(2))
    query_time = match.group(3)
    if query_time is None:
        lower = datetime.combine(query_date, time.min, tzinfo=UTC)
        upper = lower + timedelta(days=1)
    else:
        zone = match.group(4) or "Z"
        datetime_text = f"{match.group(2)}T{query_time}{zone.replace('Z', '+00:00')}"
        lower = datetime.fromisoformat(datetime_text).astimezone(UTC)
        upper = lower + timedelta(seconds=1)

    resource_date = _resource_date(resource_type, resource)
    if resource_date is None:
        return False
    actual = _parse_resource_datetime(resource_date)
    if actual is None:
        return False
    if prefix == "eq":
        return lower <= actual < upper
    if prefix == "ge":
        return actual >= lower
    if prefix == "le":
        return actual < upper
    if prefix == "gt":
        return actual >= upper
    return actual < lower


def _resource_date(resource_type: str, resource: dict[str, Any]) -> Any:
    for path in DATE_FIELDS.get(resource_type, ()):
        value: Any = resource
        for key in path:
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(key)
        if value is not None:
            return value
    return None


def _parse_resource_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            parsed = datetime.combine(date.fromisoformat(value), time.min, tzinfo=UTC)
        else:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    except ValueError:
        return None


def _sort_by_date(
    resources: list[dict[str, Any]], resource_type: str, *, descending: bool
) -> list[dict[str, Any]]:
    dated: list[tuple[datetime, dict[str, Any]]] = []
    undated: list[dict[str, Any]] = []
    for resource in resources:
        parsed = _parse_resource_datetime(_resource_date(resource_type, resource))
        if parsed is None:
            undated.append(resource)
        else:
            dated.append((parsed, resource))
    dated.sort(key=lambda pair: pair[0], reverse=descending)
    undated.sort(key=lambda resource: str(resource.get("id", "")))
    return [resource for _, resource in dated] + undated


def _has_code(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("coding"), list)
        and any(
            isinstance(item, dict) and isinstance(item.get("code"), str) for item in value["coding"]
        )
    )


def _contains_code(value: Any, code: Any) -> bool:
    if not isinstance(code, str):
        return False
    if "|" in code:
        system, expected_code = code.split("|", 1)
        expected_system = system or None
    else:
        expected_code = code
        expected_system = None

    if isinstance(value, dict):
        codings = value.get("coding")
        if isinstance(codings, list) and any(
            isinstance(coding, dict)
            and coding.get("code") == expected_code
            and (expected_system is None or coding.get("system") == expected_system)
            for coding in codings
        ):
            return True
        return any(_contains_code(item, code) for item in value.values())
    if isinstance(value, list):
        return any(_contains_code(item, code) for item in value)
    return False
