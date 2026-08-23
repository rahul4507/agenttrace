"""The ingest boundary.

Every source normalises to `Conversation` here, so nothing downstream depends on where a
conversation came from and a new source is a new adapter.

Adapters need a partial-failure policy: customer exports arrive as call-log API responses,
CRM CSVs, telephony dumps and hand-assembled JSON, and a large export always contains some
rows that cannot be parsed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ..errors import IngestError
from ..models import Conversation

log = logging.getLogger("agenttrace.ingest")


@dataclass
class IngestReport:
    """What came in, what was dropped, and why.

    Returned alongside the data rather than logged and forgotten. A coverage report built
    on a corpus that silently dropped 8% of its rows is misleading, so the drop count has
    to travel with the data all the way to the UI.
    """

    source: str
    accepted: int = 0
    rejected: int = 0
    reasons: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def reject_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0

    def summary(self) -> str:
        if not self.rejected:
            return f"{self.source}: {self.accepted} conversations, none rejected"
        top = ", ".join(f"{k} ({v})" for k, v in
                        sorted(self.reasons.items(), key=lambda kv: -kv[1])[:3])
        return (f"{self.source}: {self.accepted} accepted, {self.rejected} rejected "
                f"({self.reject_rate:.1%}) -- {top}")


@runtime_checkable
class Source(Protocol):
    """A transcript source."""

    name: str

    def load(self) -> tuple[list[Conversation], IngestReport]:
        ...


# A corpus that is mostly garbage means the adapter is wrong, not that the data is bad.
# Failing at this threshold turns a silently-skewed report into a loud, fixable error.
MAX_REJECT_RATE = 0.20


def enforce_quality(report: IngestReport, *, max_reject_rate: float = MAX_REJECT_RATE) -> None:
    if report.total and report.reject_rate > max_reject_rate:
        raise IngestError(
            f"{report.source}: rejected {report.reject_rate:.1%} of rows, above the "
            f"{max_reject_rate:.0%} threshold -- the adapter is probably mismatched to "
            f"this export, rather than the data being bad",
            context={"reasons": report.reasons, "accepted": report.accepted},
        )
