# Contributing

## Standards

- Keep text short and direct.
- Add comments only to explain why non-obvious behavior is needed.
- Describe what code does; avoid history, process, and internal names.
- Reuse shared concepts instead of declaring them more than once.
- Prefer a small number of parametrized tests through the public API.
- Use synthetic data only.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run mypy src
uv run pytest
uv run python scripts/maintenance.py check
uv run python scripts/maintenance.py tasks --check
uv run python scripts/maintenance.py report --check
uv build
```

## Browser tests

Install the development group and Chromium before running browser tests:

```sh
uv sync --group dev
uv run playwright install chromium
CHARCOT_REQUIRE_BROWSER=1 uv run pytest tests/test_ehr_web_browser.py
```

Regenerate screenshot hashes with `uv run python scripts/maintenance.py screenshots`.
