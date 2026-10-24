"""Deterministic in-memory storage for the supported FHIR R5 resource shapes."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

SUPPORTED_TYPES = {
    "Patient",
    "Observation",
    "Condition",
    "Encounter",
    "DocumentReference",
    "Claim",
}
REFERENCE_PATTERN = re.compile(r"^([A-Z][A-Za-z]+)/([A-Za-z0-9.-]{1,64})$")
REQUIRED_FIELDS = {
    "Patient": (),
    "Observation": ("status", "code", "subject"),
    "Condition": ("subject", "code"),
    "Encounter": ("status", "class"),
    "DocumentReference": ("status", "content"),
    "Claim": ("status", "type", "use", "patient"),
}


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
        filters = params or {}
        invalid_params = set(filters) - {"subject", "patient", "code"}
        if invalid_params:
            raise ValueError(f"unsupported search parameter: {sorted(invalid_params)[0]}")
        results = []
        for (kind, _), resource in sorted(self._resources.items()):
            if kind != resource_type:
                continue
            if all(self._matches_param(resource, name, value) for name, value in filters.items()):
                results.append(deepcopy(resource))
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
            field in resource for field in ("valueQuantity", "valueString", "valueInteger")
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
    def _matches_param(resource: dict[str, Any], name: str, value: Any) -> bool:
        if name in {"subject", "patient"}:
            references = [
                item["reference"]
                for field in ("subject", "patient")
                if isinstance((item := resource.get(field)), dict)
                and isinstance(item.get("reference"), str)
            ]
            return value in references or any(
                reference.rsplit("/", 1)[-1] == value for reference in references
            )
        return _contains_code(resource, value)


def _has_code(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("coding"), list)
        and any(
            isinstance(item, dict) and isinstance(item.get("code"), str) for item in value["coding"]
        )
    )


def _contains_code(value: Any, code: Any) -> bool:
    if isinstance(value, dict):
        if value.get("code") == code:
            return True
        return any(_contains_code(item, code) for item in value.values())
    if isinstance(value, list):
        return any(_contains_code(item, code) for item in value)
    return False
