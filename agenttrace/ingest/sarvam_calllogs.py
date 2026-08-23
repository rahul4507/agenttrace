"""Sarvam Voice Agents call-log source.

Reads production conversations from the platform and normalises them to `Conversation`.

Written defensively: the field names below are our reading of the call-log shape, and an
adapter against an external API drifts. Every access tolerates absence, an unrecognised
shape is rejected with a reason rather than raised, and the mapping lives in one dict so a
correction is a single change.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ..config import Settings
from ..costs import Usage
from ..errors import AuthError, IngestError, ServerError
from ..models import Conversation, Outcome, Role, ToolCall, Turn
from .base import IngestReport

log = logging.getLogger("agenttrace.ingest.sarvam")

# The platform's dispositions mapped onto our outcome vocabulary. Anything unmapped becomes
# None rather than being guessed at -- a wrong outcome silently corrupts every failure rate
# downstream, and "unknown" is a far cheaper mistake than "wrong".
_OUTCOME_MAP: dict[str, Outcome] = {
    "goal_achieved": Outcome.RESOLVED,
    "completed": Outcome.RESOLVED,
    "success": Outcome.RESOLVED,
    "partial": Outcome.PARTIAL,
    "in_progress": Outcome.PARTIAL,
    "transferred": Outcome.ESCALATED,
    "escalated": Outcome.ESCALATED,
    "user_hangup": Outcome.CALLER_ABANDONED,
    "abandoned": Outcome.CALLER_ABANDONED,
    "no_answer": Outcome.NOT_CONNECTED,
    "busy": Outcome.NOT_CONNECTED,
    "failed": Outcome.NOT_CONNECTED,
    "error": Outcome.AGENT_ERROR,
    "bot_error": Outcome.AGENT_ERROR,
}

_ROLE_MAP = {
    "assistant": Role.AGENT, "agent": Role.AGENT, "bot": Role.AGENT,
    "user": Role.CALLER, "caller": Role.CALLER, "customer": Role.CALLER, "human": Role.CALLER,
    "system": Role.SYSTEM, "tool": Role.SYSTEM,
}


class SarvamCallLogSource:
    name = "sarvam_calllogs"

    def __init__(self, settings: Settings, *, agent_id: str, limit: int = 500,
                 client: httpx.Client | None = None) -> None:
        self.settings = settings
        self.agent_id = agent_id
        self.limit = limit
        self._client = client or httpx.Client(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.request_timeout_s,
                                  connect=settings.connect_timeout_s),
        )

    def load(self) -> tuple[list[Conversation], IngestReport]:
        report = IngestReport(source=f"sarvam:{self.agent_id}")
        out: list[Conversation] = []
        for raw in self._fetch_pages():
            try:
                conv = self._normalise(raw)
            except Exception as exc:               # one bad row must not kill the import
                report.reject(f"{type(exc).__name__}: {str(exc)[:50]}")
                continue
            if conv is None:
                report.reject("unrecognised call-log shape")
                continue
            if not conv.turns:
                report.reject("empty transcript")
                continue
            out.append(conv)
            report.accepted += 1
        # Deliberately not calling enforce_quality here: on a live API a high reject rate
        # means our field mapping has drifted, which we want visible in the report and
        # investigated -- not turned into an exception that hides the rows that did work.
        if report.reject_rate > 0.2:
            log.error("sarvam call-log adapter rejected %.0f%% of rows -- field mapping has "
                      "probably drifted: %s", 100 * report.reject_rate, report.reasons)
        return out, report

    def _fetch_pages(self):
        """Paginate the call-log endpoint, yielding raw dicts."""
        headers = {"api-subscription-key": self.settings.require_key()}
        fetched, page = 0, 1
        while fetched < self.limit:
            try:
                r = self._client.get(
                    "/v1/call-logs",
                    params={"agent_id": self.agent_id, "page": page,
                            "page_size": min(100, self.limit - fetched)},
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise IngestError(f"call-log fetch failed: {type(exc).__name__}") from exc
            if r.status_code in (401, 403):
                raise AuthError("call-log access denied for this key", status=r.status_code)
            if r.status_code == 404:
                raise IngestError(
                    "call-log endpoint not available for this account. Voice Agents "
                    "call-log access may need to be enabled -- fall back to the JSONL "
                    "adapter with an exported corpus.")
            if r.status_code >= 500:
                raise ServerError("call-log endpoint error", status=r.status_code)
            body = r.json()
            items = body.get("items") or body.get("data") or body.get("call_logs") or []
            if not items:
                return
            yield from items
            fetched += len(items)
            page += 1
            if not body.get("has_more", len(items) >= 100):
                return

    def _normalise(self, raw: dict[str, Any]) -> Conversation | None:
        turns_raw = (raw.get("transcript") or raw.get("turns")
                     or raw.get("messages") or raw.get("conversation") or [])
        if not isinstance(turns_raw, list):
            return None

        turns: list[Turn] = []
        for t in turns_raw:
            if not isinstance(t, dict):
                continue
            role = _ROLE_MAP.get(str(t.get("role") or t.get("speaker") or "").lower())
            text = t.get("text") or t.get("content") or t.get("message") or ""
            if role is None or not str(text).strip():
                continue
            tool_calls = [
                ToolCall(name=str(tc.get("name") or tc.get("tool") or "unknown"),
                         arguments=tc.get("arguments") or tc.get("args") or {},
                         ok=bool(tc.get("ok", tc.get("success", True))),
                         latency_ms=tc.get("latency_ms"),
                         error=tc.get("error"))
                for tc in (t.get("tool_calls") or t.get("tools") or [])
                if isinstance(tc, dict)
            ]
            turns.append(Turn(
                role=role, text=str(text),
                language=t.get("language") or t.get("lang"),
                offset_ms=t.get("offset_ms") or t.get("start_ms"),
                duration_ms=t.get("duration_ms"),
                asr_confidence=t.get("confidence") or t.get("asr_confidence"),
                tool_calls=tool_calls,
                barge_in=bool(t.get("interrupted") or t.get("barge_in") or False),
            ))

        call_id = raw.get("id") or raw.get("call_id") or raw.get("conversation_id")
        if not call_id:
            return None

        disposition = str(raw.get("disposition") or raw.get("status")
                          or raw.get("end_reason") or "").lower()
        usage_raw = raw.get("usage") or {}
        usage = Usage(
            audio_seconds=float(raw.get("duration_seconds") or raw.get("duration") or 0.0),
            tts_characters=int(usage_raw.get("tts_characters", 0) or 0),
            llm_input_tokens=int(usage_raw.get("llm_input_tokens", 0) or 0),
            llm_cached_input_tokens=int(usage_raw.get("llm_cached_tokens", 0) or 0),
            llm_output_tokens=int(usage_raw.get("llm_output_tokens", 0) or 0),
            telephony_seconds=float(raw.get("duration_seconds") or 0.0),
            turns=len(turns),
        ) if usage_raw or raw.get("duration_seconds") else None

        return Conversation(
            id=str(call_id),
            agent_id=str(raw.get("agent_id") or self.agent_id),
            # Version is the axis regression attribution pivots on. If the platform does
            # not report one, say so explicitly rather than defaulting to "v1" -- a fake
            # version makes every version comparison silently meaningless.
            agent_version=str(raw.get("agent_version") or raw.get("version") or "unversioned"),
            channel=str(raw.get("channel") or "telephony"),
            language=str(raw.get("language") or "hi-IN"),
            started_at=raw.get("started_at") or raw.get("created_at"),
            duration_s=float(raw.get("duration_seconds") or raw.get("duration") or 0.0),
            turns=turns,
            outcome=_OUTCOME_MAP.get(disposition),
            disposition=disposition or None,
            usage=usage,
            metadata=dict(raw.get("metadata") or {}),
            source=self.name,
        )
