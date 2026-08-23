"""Exception hierarchy.

Errors carry their own retry semantics (`retryable`, `retry_after_s`) so callers do not
have to match on messages. Anything escaping this package that is not an AgentTraceError
is a bug.
"""

from __future__ import annotations


class AgentTraceError(Exception):
    """Base class for errors this package raises."""

    retryable: bool = False
    retry_after_s: float | None = None

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        # For structured logs. Must not contain secrets or unredacted PII.
        self.context = context or {}

    def __str__(self) -> str:
        if not self.context:
            return self.message
        extras = " ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))
        return f"{self.message} [{extras}]"


# Configuration and programmer error

class ConfigError(AgentTraceError):
    """Missing or malformed configuration. Raised at startup."""


class SuiteError(AgentTraceError):
    """A scenario file is invalid."""


class IngestError(AgentTraceError):
    """A transcript source produced something we cannot normalise."""


# Upstream API failures

class UpstreamError(AgentTraceError):
    """Anything originating from the Sarvam API."""

    def __init__(self, message: str, *, status: int | None = None, context=None) -> None:
        super().__init__(message, context={**(context or {}), "status": status})
        self.status = status


class AuthError(UpstreamError):
    """401/403. Not retryable with the same credentials."""


class BadRequestError(UpstreamError):
    """4xx caused by our payload: malformed body, unsupported language, oversized input."""


class RateLimitError(UpstreamError):
    """429. Retryable, but only after the server-provided delay."""

    retryable = True

    def __init__(self, message: str, *, retry_after_s: float | None = None, **kw) -> None:
        super().__init__(message, **kw)
        self.retry_after_s = retry_after_s


class ServerError(UpstreamError):
    """5xx. Retryable with backoff."""

    retryable = True


class TruncatedResponseError(UpstreamError):
    """HTTP 200 with empty content because the token budget was exhausted.

    Sarvam-105B emits `reasoning_content` before `content`, so a small max_tokens yields
    a billed response with no answer. Distinct type so the failure surfaces here rather
    than downstream as a schema error. Not retryable as-is; the client escalates
    max_tokens once.
    """

    retryable = False

    def __init__(self, message: str, *, max_tokens: int, finish_reason: str | None = None,
                 reasoning_tokens: int = 0, **kw) -> None:
        ctx = dict(kw.pop("context", None) or {})
        ctx.update(max_tokens=max_tokens, finish_reason=finish_reason,
                   reasoning_tokens=reasoning_tokens)
        super().__init__(message, context=ctx, **kw)
        self.max_tokens = max_tokens
        self.finish_reason = finish_reason
        self.reasoning_tokens = reasoning_tokens


class TimeoutError_(UpstreamError):
    """Request exceeded its deadline. Retryable, but counts toward the circuit breaker."""

    retryable = True


# Guardrails

class GuardrailError(AgentTraceError):
    """A self-imposed limit fired. Not an upstream failure."""


class BudgetExceededError(GuardrailError):
    """The run would exceed its rupee ceiling. Hard stop; never retried."""

    def __init__(self, message: str, *, spent_inr: float, limit_inr: float, **kw) -> None:
        ctx = dict(kw.pop("context", None) or {})
        ctx.update(spent_inr=round(spent_inr, 4), limit_inr=round(limit_inr, 4))
        super().__init__(message, context=ctx)
        self.spent_inr = spent_inr
        self.limit_inr = limit_inr


class CircuitOpenError(GuardrailError):
    """Upstream is unhealthy; requests are being shed."""

    retryable = True

    def __init__(self, message: str, *, reopens_in_s: float, **kw) -> None:
        super().__init__(message, **kw)
        self.retry_after_s = reopens_in_s
        self.reopens_in_s = reopens_in_s


class StructuredOutputError(AgentTraceError):
    """Model output did not satisfy the requested schema.

    Worth one repair attempt; a second failure indicates a bad prompt or schema.
    """

    retryable = True

    def __init__(self, message: str, *, raw: str | None = None, **kw) -> None:
        super().__init__(message, **kw)
        # Truncated: raw model output can be huge, and can echo caller PII.
        self.raw = (raw or "")[:2000]
