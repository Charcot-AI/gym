"""Private task rendering and composition options."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from charcot_gym.synth.hard_multimodal import RenderedPage


class PageRenderer(Protocol):
    """Render one document page."""

    def __call__(  # noqa: PLR0913
        self,
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
        """Return a rendered page."""


@dataclass(frozen=True)
class PageLayout:
    """Named page renderer."""

    name: str
    render: PageRenderer


@dataclass(frozen=True)
class Distribution:
    """Private task composition and rendering choices."""

    page_layouts: Mapping[str, tuple[PageLayout, ...]]
    document_trap_pairs: tuple[tuple[str, str], ...] = ()
    claim_pack_cycle: tuple[tuple[str, ...], ...] = ()
