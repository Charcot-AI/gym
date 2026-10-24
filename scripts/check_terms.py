from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = (
    "v" + "oa",
    "v" + "oa " + "health",
    "bio" + "stack",
    "m" + "cl",
    "t" + "iss",
    "t" + "uss",
    "uni" + "med",
    "sig" + "tap",
    "c" + "pt",
    "sno" + "med",
    "x12 " + "carc",
)
PATTERNS = [re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE) for term in BLOCKED]
TICKET_PREFIXES = ("a" + "g", "d" + "l")
TICKET_PATTERN = re.compile(
    r"\b(?:" + "|".join(TICKET_PREFIXES) + ")" + re.escape("#"),
    re.IGNORECASE,
)
SKIP_DIRECTORIES = {".git", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "__pycache__"}


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRECTORIES for part in path.parts):
            continue
        relative_path = str(path.relative_to(ROOT))
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for line_number, line in enumerate(content.splitlines(), start=1):
            searchable = f"{relative_path}\n{line}"
            if any(pattern.search(searchable) for pattern in PATTERNS) or TICKET_PATTERN.search(
                searchable
            ):
                findings.append(f"{relative_path}:{line_number}")
    if findings:
        print("Disallowed terms detected:")
        print("\n".join(findings))
        return 1
    print("Term check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
