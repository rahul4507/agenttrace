"""Canonical domain model.

Every source is normalised into these types at the ingest boundary; nothing downstream
depends on where a conversation came from.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .costs import Usage


class Role(StrEnum):
    AGENT = "agent"
    CALLER = "caller"
    SYSTEM = "system"


class Outcome(StrEnum):
    """How a conversation ended.

    Not a boolean: `resolved` and `escalated` are both acceptable endings, while
    `caller_abandoned` and `agent_error` have different owners.
    """

    RESOLVED = "resolved"                 # goal achieved (e.g. promise-to-pay captured)
    PARTIAL = "partial"                   # progress made, goal not reached
    ESCALATED = "escalated"               # correctly handed to a human
    CALLER_ABANDONED = "caller_abandoned"  # caller hung up mid-flow
    NOT_CONNECTED = "not_connected"       # never reached a human at all
    AGENT_ERROR = "agent_error"           # our fault: loop, wrong info, tool failure
    COMPLIANCE_BREACH = "compliance_breach"  # said something it must not

    @property
    def is_failure(self) -> bool:
        """Agent-attributable failure only. A caller hanging up is not an agent failure."""
        return self in {Outcome.AGENT_ERROR, Outcome.COMPLIANCE_BREACH}

    @property
    def is_success(self) -> bool:
        return self in {Outcome.RESOLVED, Outcome.ESCALATED}


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    latency_ms: int | None = None
    error: str | None = None


class Turn(BaseModel):
    role: Role
    text: str
    language: str | None = None          # BCP-47-ish: "hi-IN", "en-IN", "hi-en" for code-mix
    offset_ms: int | None = None         # from call start
    duration_ms: int | None = None
    asr_confidence: float | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    barge_in: bool = False               # caller interrupted the agent

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


class Conversation(BaseModel):
    """One normalised call or chat. The unit of analysis."""

    id: str
    agent_id: str
    agent_version: str                   # pivot for regression attribution
    channel: str = "telephony"           # telephony | whatsapp | web | api
    language: str = "hi-IN"
    started_at: datetime | None = None
    duration_s: float = 0.0
    turns: list[Turn] = Field(default_factory=list)
    outcome: Outcome | None = None       # as reported by the source, if it reports one
    disposition: str | None = None       # the source's own free-text label
    usage: Usage | None = None           # measured usage, when the source provides it
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str = "unknown"              # which adapter produced this

    model_config = {"arbitrary_types_allowed": True}

    # Derived views

    @property
    def turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role in {Role.AGENT, Role.CALLER})

    @property
    def agent_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.role is Role.AGENT)

    @property
    def caller_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.role is Role.CALLER)

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for t in self.turns for tc in t.tool_calls]

    def called_tool(self, name: str) -> bool:
        return any(tc.name == name for tc in self.tool_calls)

    def transcript(self, *, redacted: bool = True, max_chars: int | None = None) -> str:
        """Readable transcript.

        Redacted by default: this string is sent to the model, cached, and rendered in the
        dashboard.
        """
        from .redact import redact  # local import avoids a cycle

        lines = []
        for t in self.turns:
            if t.role is Role.SYSTEM:
                continue
            who = "AGENT" if t.role is Role.AGENT else "CALLER"
            text = redact(t.text) if redacted else t.text
            lines.append(f"{who}: {text}")
        out = "\n".join(lines)
        return out if max_chars is None else out[:max_chars]

    def estimated_usage(self) -> Usage:
        """Estimate usage when the source reports none.

        Callers must surface that the result is estimated rather than measured.
        """
        if self.usage is not None:
            return self.usage
        agent_chars = len(self.agent_text)
        # ~4 chars/token for Indic-Latin mixed script.
        out_tokens = max(1, agent_chars // 4)
        # Each turn resends system prompt + history; most is cache-eligible after turn one.
        base_prompt_tokens = 1400
        turns = max(1, self.turn_count)
        cached = base_prompt_tokens * max(0, turns - 1)
        fresh = base_prompt_tokens + len(self.caller_text) // 4
        return Usage(
            audio_seconds=self.duration_s,
            tts_characters=agent_chars,
            llm_input_tokens=fresh,
            llm_cached_input_tokens=cached,
            llm_output_tokens=out_tokens,
            telephony_seconds=self.duration_s,
            turns=turns,
        )
