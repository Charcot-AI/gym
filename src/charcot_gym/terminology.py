"""Search the bundled laboratory terminology subset."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from importlib.resources import files
from typing import Any

TOKEN_PATTERN = re.compile(r"[\w*]+(?:[-/][\w*]+)*", re.UNICODE)


def terminology_search(system: str, query: str, limit: int = 10) -> list[dict[str, str]]:
    """Return terminology entries whose display or synonyms contain query tokens."""
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    query_tokens = _tokens(query)
    if not query_tokens:
        return []
    matches = []
    for entry in _entries():
        if entry["system"] != system:
            continue
        searchable = " ".join([entry["display"], *entry["synonyms"]])
        if query_tokens <= _tokens(searchable):
            matches.append(
                {
                    "system": entry["system"],
                    "code": entry["code"],
                    "display": entry["display"],
                    "unit": entry["unit"],
                }
            )
    return matches[:limit]


@lru_cache(maxsize=1)
def _entries() -> tuple[dict[str, Any], ...]:
    """Read the bundled subset once per process."""
    content = files("charcot_gym").joinpath("data", "loinc_subset.json").read_text(encoding="utf-8")
    entries = json.loads(content)
    if not isinstance(entries, list):
        raise ValueError("terminology data must be a list")
    return tuple(entries)


def _tokens(value: str) -> set[str]:
    tokens = set()
    for token in TOKEN_PATTERN.findall(value.casefold()):
        tokens.add(token)
        tokens.update(part for part in re.split(r"[-/]+", token) if part)
    return tokens
