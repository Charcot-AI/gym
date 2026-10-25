from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageDraw

from charcot_gym.synth.distribution import Distribution, PageLayout
from charcot_gym.synth.hard_claims import generate_hard_claim_task_sets
from charcot_gym.synth.hard_document_fhir import generate_hard_document_fhir_task_sets
from charcot_gym.synth.hard_multimodal import (
    RenderedPage,
    prepare_multimodal_task,
    render_page,
)
from charcot_gym.synth.hard_multitool import prepare_hard_task


def _distinct_renderer(  # noqa: PLR0913
    text: str,
    *,
    received_at: str,
    page_number: int,
    page_count: int,
    sender_fax: str,
    fax: bool,
    seed: int,
    footer_issuer: str | None = None,
) -> RenderedPage:
    page = render_page(
        text,
        received_at=received_at,
        page_number=page_number,
        page_count=page_count,
        sender_fax=sender_fax,
        fax=fax,
        seed=seed,
        footer_issuer=footer_issuer,
    )
    image = Image.open(io.BytesIO(page.png)).convert("L")
    ImageDraw.Draw(image).rectangle((0, 0, 20, 20), fill=0)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return RenderedPage(
        output.getvalue(),
        page.text_boxes,
        page.header_box,
        page.stamp_box,
        page.table_row_drifts,
    )


def _private_image_path(root: Path, task: dict[str, object]) -> Path:
    reference = task["environment"]["inbox"]["documents"][0]["pages"][0]
    return root / reference


def test_private_distribution_does_not_change_public_generators() -> None:
    default_public, _default_private = generate_hard_document_fhir_task_sets(1234)
    document_distribution = Distribution(
        page_layouts={},
        document_trap_pairs=(("corrected_report", "wrong_patient"),),
    )
    distributed_public, distributed_private = generate_hard_document_fhir_task_sets(
        1234, distribution=document_distribution
    )
    assert distributed_public == default_public
    assert distributed_private != _default_private

    default_claims, _default_private_claims = generate_hard_claim_task_sets(1234)
    claims_distribution = Distribution(
        page_layouts={},
        claim_pack_cycle=(("pa_insufficient",),),
    )
    distributed_claims, distributed_private_claims = generate_hard_claim_task_sets(
        1234, distribution=claims_distribution
    )
    assert distributed_claims == default_claims
    assert distributed_private_claims != _default_private_claims


def test_private_layout_metadata_and_rendering_are_applied(tmp_path: Path) -> None:
    _public, private = generate_hard_document_fhir_task_sets(1234)
    source = prepare_hard_task(private[0])
    title = source["environment"]["inbox"]["documents"][0]["title"]
    distribution = Distribution(
        page_layouts={title: (PageLayout("custom-layout", _distinct_renderer),)}
    )
    default_root = tmp_path / "default"
    private_root = tmp_path / "private"
    default_task = prepare_multimodal_task(
        source,
        default_root,
        check=False,
        private_tasks_dir=default_root / "private" / "tasks",
    )
    distributed_task = prepare_multimodal_task(
        source,
        private_root,
        check=False,
        distribution=distribution,
        private_tasks_dir=private_root / "private" / "tasks",
    )
    assert "custom-layout" in distributed_task["metadata"]["page_layouts"].values()
    assert (
        _private_image_path(default_root, default_task).read_bytes()
        != _private_image_path(private_root, distributed_task).read_bytes()
    )
