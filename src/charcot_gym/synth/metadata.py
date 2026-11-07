"""Shared metadata normalization for generated tasks."""

from __future__ import annotations

from typing import Any


def with_variant_group(task: dict[str, Any]) -> dict[str, Any]:
    """Return a task with a stable group shared by its scenario variants."""
    metadata = dict(task.get("metadata") or {})
    scenario = (
        metadata.get("scenario_id")
        or metadata.get("template_id")
        or metadata.get("template")
        or metadata.get("script")
        or metadata.get("scenario")
        or task.get("id")
    )
    if isinstance(scenario, dict):
        scenario = scenario.get("id") or scenario.get("name") or task.get("id")
    metadata["variant_group"] = f"{task['family']}:{scenario}"
    return task | {"metadata": metadata}
