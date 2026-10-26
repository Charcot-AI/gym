import importlib.util
import sys
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "gym_maintenance",
    Path(__file__).resolve().parents[1] / "scripts" / "maintenance.py",
)
assert spec is not None and spec.loader is not None
maintenance = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = maintenance
spec.loader.exec_module(maintenance)


def test_task_count_table_uses_display_names_and_ids():
    public_counts = Counter(
        {
            ("claims", "standard", "text"): 1,
            ("clinical_coding", "standard", "text"): 1,
            ("document_fhir", "standard", "text"): 1,
        }
    )

    table = "\n".join(maintenance._task_count_table(public_counts, Counter()))

    assert "Claims | `claims`" in table
    assert "Clinical coding | `clinical_coding`" in table
    assert "Document → FHIR | `document_fhir`" in table


def test_report_checks_readme_task_table(tmp_path, monkeypatch):
    task_table = (
        "| Family | Family ID | Tier | Modality | Public | Private |\n"
        "| --- | --- | --- | --- | ---: | ---: |"
    )
    monkeypatch.setattr(maintenance, "ROOT", tmp_path)
    monkeypatch.setattr(
        maintenance,
        "_quality_report_artifacts",
        lambda: ("# Quality report\n", task_table),
    )
    (tmp_path / "README.md").write_text(
        "## Task coverage\n\n<!-- tasks:start -->\nstale table\n<!-- tasks:end -->\n",
        encoding="utf-8",
    )

    assert maintenance.quality_report() == 0
    readme = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert f"<!-- tasks:start -->\n{task_table}\n<!-- tasks:end -->" in readme
    assert maintenance.quality_report(check=True) == 0

    (tmp_path / "README.md").write_text(readme.replace(task_table, "stale table"), encoding="utf-8")
    assert maintenance.quality_report(check=True) == 1
