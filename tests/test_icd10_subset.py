from __future__ import annotations

import json
from pathlib import Path


def test_medication_only_condition_code_is_from_cms_subset() -> None:
    path = Path(__file__).parents[1] / "src" / "charcot_gym" / "data" / "icd10cm_subset.json"
    entries = json.loads(path.read_text(encoding="utf-8"))
    code = next(item for item in entries if item["code"] == "E03.9")
    assert code["billable"] is True
    assert code["display"] == "Hypothyroidism, unspecified"
    assert code["cms_source_url"].startswith("https://www.cms.gov/")
