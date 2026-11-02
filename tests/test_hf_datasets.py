from __future__ import annotations

import base64
import contextlib
import importlib.util
import io
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import datasets
import httpx
import pytest

from charcot_gym import Task, load_tasks
from charcot_gym.runner.episode import _image_data_uri

spec = importlib.util.spec_from_file_location(
    "gym_hf_datasets",
    Path(__file__).resolve().parents[1] / "scripts" / "hf_datasets.py",
)
assert spec is not None and spec.loader is not None
hf_datasets = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = hf_datasets
spec.loader.exec_module(hf_datasets)

PUBLIC_TASKS = [task for task in load_tasks() if task.split == "public"]


def _image_bytes(image: Any) -> bytes:
    if isinstance(image, dict) and image["bytes"] is not None:
        return image["bytes"]
    if isinstance(image, bytes):
        return image
    if hasattr(image, "getvalue"):
        return image.getvalue()
    raise AssertionError(f"Unexpected image value: {type(image)!r}")


def _write_payload(
    directory: Path,
    *,
    data: bytes = b"data",
    readme: bytes = b"readme",
    logo: bytes = b"logo",
) -> None:
    (directory / "data").mkdir(parents=True, exist_ok=True)
    (directory / "assets").mkdir(parents=True, exist_ok=True)
    (directory / "data" / "public.parquet").write_bytes(data)
    (directory / "README.md").write_bytes(readme)
    (directory / "assets" / "logo.png").write_bytes(logo)


def _mock_existing_dataset_tag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    change: str,
) -> list[dict[str, Any]]:
    tag = f"v{hf_datasets.project_version()}"
    remote_dir = tmp_path / "remote"
    _write_payload(
        remote_dir,
        data=b"other data" if change == "data" else b"data",
        readme=b"old readme" if change == "readme" else b"readme",
        logo=b"old logo" if change == "assets" else b"logo",
    )

    class FakeApi:
        def __init__(self, token: str) -> None:
            assert token == "test-token"

        def list_repo_refs(self, repo_id: str, *, repo_type: str) -> Any:
            return SimpleNamespace(tags=[SimpleNamespace(name=tag)])

    def fake_write_dataset(
        tasks: list[Task],
        out_dir: Path,
        *,
        split_name: str,
        tasks_dir: Path | None,
    ) -> None:
        _write_payload(out_dir)

    uploads: list[dict[str, Any]] = []
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hf_datasets, "load_tasks", lambda: [])
    monkeypatch.setattr(hf_datasets, "write_dataset", fake_write_dataset)
    monkeypatch.setattr(hf_datasets, "HfApi", FakeApi)
    monkeypatch.setattr(hf_datasets, "create_repo", lambda *args, **kwargs: None)
    monkeypatch.setattr(hf_datasets, "snapshot_download", lambda *args, **kwargs: remote_dir)
    monkeypatch.setattr(
        hf_datasets,
        "upload_folder",
        lambda **kwargs: uploads.append(kwargs),
    )
    return uploads


def test_columns_follow_task_model_fields() -> None:
    assert [column.name for column in hf_datasets.COLUMNS] == [*Task.model_fields, "images"]
    assert "clinical_reasoning" in hf_datasets.COLUMNS[1].description


def test_public_dataset_round_trip_and_oracle(tmp_path: Path) -> None:
    hf_datasets.write_dataset(PUBLIC_TASKS, tmp_path, split_name="public", tasks_dir=None)
    parquet_path = tmp_path / "data" / "public-00000-of-00001.parquet"
    exported = datasets.load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    ).cast_column("images", datasets.Sequence(datasets.Image(decode=False)))
    originals = {task.id: task for task in PUBLIC_TASKS}
    rebuilt = {row["id"]: hf_datasets.row_task(row) for row in exported}

    assert rebuilt == originals
    for task_id, task in rebuilt.items():
        if task.modality == "text" and task.family != "smoke":
            assert hf_datasets.oracle_score(task) == 1.0, task_id

    image_rows = [row for row in exported if row["modality"] == "image"]
    assert image_rows
    assert (tmp_path / "assets" / "charcot-logo-black.png").is_file()
    for row in image_rows:
        task = originals[row["id"]]
        expected = [
            hf_datasets.read_task_image(reference, None)["bytes"]
            for reference in hf_datasets._image_references(task)
        ]
        assert [_image_bytes(image) for image in row["images"]] == expected


def test_parquet_image_bytes_match_runner_data_uri(tmp_path: Path) -> None:
    task = next(task for task in PUBLIC_TASKS if task.modality == "image")
    reference = hf_datasets._image_references(task)[0]
    parquet_path = tmp_path / "public.parquet"
    hf_datasets.write_task_parquet([task], parquet_path)
    exported = datasets.load_dataset(
        "parquet",
        data_files=str(parquet_path),
        split="train",
    ).cast_column("images", datasets.Sequence(datasets.Image(decode=False)))

    expected = base64.b64decode(_image_data_uri(reference).partition(",")[2])
    assert _image_bytes(exported[0]["images"][0]) == expected


def test_rendered_card_quick_start_runs_locally(tmp_path: Path) -> None:
    task = next(task for task in PUBLIC_TASKS if task.modality == "text")
    hf_datasets.write_dataset([task], tmp_path, split_name="public", tasks_dir=None)
    card = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "\nprivate:" not in card.split("---", maxsplit=2)[1]
    code = re.search(r"```python\n(.*?)\n```", card, re.DOTALL)
    assert code is not None
    snippet = code.group(1).replace(
        (
            f'load_dataset("charcot-ai/gym", split="public", '
            f'revision="v{hf_datasets.project_version()}")'
        ),
        'load_dataset("parquet", data_files={"train": str(parquet_path)}, split="train")',
    )
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(snippet, {"parquet_path": tmp_path / "data" / "public-00000-of-00001.parquet"})
    assert output.getvalue().strip() == "1.0"


@pytest.mark.parametrize("change", ["none", "readme", "assets", "data"])
def test_push_existing_tag_keeps_data_immutable(
    change: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    uploads = _mock_existing_dataset_tag(monkeypatch, tmp_path, change=change)

    if change == "data":
        with pytest.raises(ValueError, match="bump the version"):
            hf_datasets.push_dataset("charcot-ai/gym", public=False)
        assert not uploads
        return

    assert hf_datasets.push_dataset("charcot-ai/gym", public=False) == 0
    if change == "none":
        assert not uploads
        assert "already matches" in capsys.readouterr().out
    else:
        assert len(uploads) == 1
        assert uploads[0]["revision"] == "main"
        assert uploads[0]["allow_patterns"] == ["README.md", "assets/*"]
        assert uploads[0]["delete_patterns"] == ["assets/*"]
        assert "Updated charcot-ai/gym@main" in capsys.readouterr().out


def test_wait_for_viewer_retries_server_errors_and_missing_viewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = httpx.Request("GET", "https://datasets-server.huggingface.co/is-valid")
    responses = iter(
        [
            httpx.Response(500, text="busy", request=request),
            httpx.Response(200, json={"status": "pending"}, request=request),
            httpx.Response(200, json={"viewer": False, "pending": ["default"]}, request=request),
            httpx.Response(200, json={"viewer": True}, request=request),
        ]
    )
    calls: list[dict[str, Any]] = []
    waits: list[float] = []

    def fake_get(url: str, **kwargs: Any) -> httpx.Response:
        calls.append({"url": url, **kwargs})
        return next(responses)

    monkeypatch.setattr(httpx, "get", fake_get)
    monkeypatch.setattr(time, "sleep", waits.append)
    hf_datasets.wait_for_viewer("charcot-ai/gym", "test-token")

    assert len(calls) == 4
    assert calls[0]["params"] == {"dataset": "charcot-ai/gym"}
    assert calls[0]["headers"] == {"Authorization": "Bearer test-token"}
    assert waits == [15.0, 15.0, 15.0]


def test_verify_rejects_card_that_differs_from_generated_card(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stale_readme = tmp_path / "README.md"
    stale_readme.write_text("stale\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hf_datasets, "load_tasks", lambda: [])
    monkeypatch.setattr(datasets, "load_dataset", lambda *args, **kwargs: [])
    monkeypatch.setattr(hf_datasets, "render_card", lambda *args, **kwargs: "current\n")
    monkeypatch.setattr(
        hf_datasets,
        "hf_hub_download",
        lambda *args, **kwargs: stale_readme,
    )

    with pytest.raises(ValueError, match="main dataset card differs"):
        hf_datasets.verify_dataset("charcot-ai/gym", f"v{hf_datasets.project_version()}")
