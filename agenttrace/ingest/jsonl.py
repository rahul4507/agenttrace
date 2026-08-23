"""JSONL transcript source.

One conversation per line: the format to ask a customer for when their export tooling is
limited, and what the fixtures use.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError

from ..models import Conversation
from .base import IngestReport, enforce_quality


class JsonlSource:
    name = "jsonl"

    def __init__(self, path: Path, *, strict: bool = False) -> None:
        self.path = Path(path)
        # strict=True re-raises on the first bad row. Useful in CI; wrong for a customer
        # export, where you want the 49,993 good rows and a report on the other 7.
        self.strict = strict

    def load(self) -> tuple[list[Conversation], IngestReport]:
        report = IngestReport(source=f"jsonl:{self.path.name}")
        out: list[Conversation] = []
        if not self.path.exists():
            from ..errors import IngestError
            raise IngestError(f"transcript file not found: {self.path}")

        with self.path.open(encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                except json.JSONDecodeError:
                    if self.strict:
                        raise
                    report.reject(f"malformed json (line {lineno})" if report.rejected < 3
                                  else "malformed json")
                    continue
                try:
                    conv = Conversation.model_validate({**raw, "source": self.name})
                except ValidationError as exc:
                    if self.strict:
                        raise
                    report.reject(exc.errors()[0]["msg"][:60])
                    continue
                if not conv.turns:
                    # An empty transcript is a real production artefact (call connected,
                    # nobody spoke) but it carries no situation, so it cannot be labelled.
                    report.reject("empty transcript")
                    continue
                out.append(conv)
                report.accepted += 1

        enforce_quality(report)
        return out, report
