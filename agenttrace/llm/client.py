"""Sarvam chat client.

The only module that performs network I/O. Handles, in one place:

    classify   HTTP status -> typed error carrying its own retry semantics
    retry      bounded attempts, exponential backoff with full jitter, honours Retry-After
    shed       circuit breaker, so retries do not amplify a sustained outage
    cap        budget check before each call, actual usage recorded after
    memoise    content-addressed cache keyed on prompt version and model
    validate   structured output against a schema, with one repair attempt
    observe    PII-redacted structured log line per attempt

Full jitter rather than plain exponential backoff: plain backoff synchronises workers onto
the same retry instant, so a recovering service receives the whole fleet at once.

Retryability belongs to the error type, not the call site. A 401 cannot succeed on retry
and consumes rate limit; a 400 would resend the same invalid payload.

Schema violations get one repair attempt. A second failure indicates the prompt or schema
is wrong, so further retries cost money without changing the outcome.
"""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..errors import (
    AgentTraceError,
    AuthError,
    BadRequestError,
    CircuitOpenError,
    RateLimitError,
    ServerError,
    StructuredOutputError,
    TimeoutError_,
    TruncatedResponseError,
    UpstreamError,
)
from ..redact import redact
from .budget import BudgetGuard
from .cache import ResponseCache
from .circuit import CircuitBreaker

log = logging.getLogger("agenttrace.llm")

T = TypeVar("T", bound=BaseModel)


# Sarvam-105B reasons before answering, so a call needs headroom for reasoning and answer.
DEFAULT_MAX_TOKENS = 3000
MAX_TOKENS_CEILING = 12000


@dataclass
class ChatResult:
    text: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    cost_inr: float = 0.0
    model: str = ""
    attempts: int = 1
    from_cache: bool = False
    # Reasoning tokens bill as output (Rs 73.2/1M) and dominate labeling cost, so they are
    # tracked separately rather than folded into output_tokens.
    reasoning_tokens: int = 0
    finish_reason: str | None = None


def _classify(status: int, body: str, headers: httpx.Headers | None = None) -> UpstreamError:
    """Map an HTTP response to a typed error carrying the retry decision."""
    snippet = redact(body)[:400]
    if status in (401, 403):
        return AuthError("Sarvam rejected the credentials", status=status,
                         context={"body": snippet})
    if status == 429:
        retry_after = None
        if headers is not None:
            raw = headers.get("retry-after")
            if raw:
                try:
                    retry_after = float(raw)
                except ValueError:
                    retry_after = None
        return RateLimitError("rate limited by Sarvam", status=status,
                              retry_after_s=retry_after, context={"body": snippet})
    if 400 <= status < 500:
        return BadRequestError("Sarvam rejected the request", status=status,
                               context={"body": snippet})
    return ServerError("Sarvam server error", status=status, context={"body": snippet})


class SarvamChatClient:
    """Chat-completions client.

    Synchronous: concurrency comes from a thread pool bounded by
    `settings.max_concurrency`, which keeps requests under the documented rate limit. The
    bottleneck is that limit rather than the event loop.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        budget: BudgetGuard | None = None,
        cache: ResponseCache | None = None,
        breaker: CircuitBreaker | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self.budget = budget or BudgetGuard(settings.run_budget_inr, name="labeling-run")
        self.cache = cache or ResponseCache(settings.cache_dir / "responses.db",
                                            namespace="chat")
        self.breaker = breaker or CircuitBreaker(
            fail_threshold=settings.circuit_fail_threshold,
            reset_timeout_s=settings.circuit_reset_s,
        )
        # Injected so tests can drive retries without sleeping and simulate 429s/500s.
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=settings.base_url,
            timeout=httpx.Timeout(settings.request_timeout_s,
                                  connect=settings.connect_timeout_s),
            transport=transport,
        )

    # --- public API ------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        prompt_version: str = "v1",
        use_cache: bool = True,
        _escalated: bool = False,
    ) -> ChatResult:
        model = model or self.settings.label_model
        key = ResponseCache.make_key("chat", model, temperature, max_tokens,
                                     prompt_version, messages)

        if use_cache and (hit := self.cache.get(key)) is not None:
            return ChatResult(**hit, from_cache=True)

        if self.settings.offline:
            raise AgentTraceError("offline mode: refusing to make a network call",
                              context={"model": model})

        self.settings.require_key()

        # Estimate before spending, assuming full max_tokens output. An early stop resumes
        # from cache; an overspend cannot be undone.
        est_in = sum(len(m.get("content", "")) for m in messages) // 4
        est = self.budget.estimate_inr(input_tokens=est_in, output_tokens=max_tokens)
        self.budget.check(est)

        try:
            result = self._chat_with_retries(messages, model, temperature, max_tokens)
        except TruncatedResponseError as exc:
            # Escalate once. A second truncation means the request is unbounded, so
            # retrying larger only costs more.
            self.budget.record(BudgetGuard.estimate_inr(input_tokens=est_in,
                                                        output_tokens=max_tokens))
            bigger = min(max_tokens * 3, MAX_TOKENS_CEILING)
            if _escalated or bigger <= max_tokens:
                raise
            log.warning("response truncated at max_tokens=%d after %d reasoning tokens; "
                        "escalating to %d", max_tokens, exc.reasoning_tokens, bigger)
            return self.chat(messages, model=model, temperature=temperature,
                             max_tokens=bigger, prompt_version=prompt_version,
                             use_cache=use_cache, _escalated=True)
        self.budget.record(result.cost_inr)

        if use_cache:
            payload = {k: v for k, v in vars(result).items() if k != "from_cache"}
            self.cache.put(key, payload)
        return result

    def structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        model: str | None = None,
        prompt_version: str = "v1",
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> tuple[T, ChatResult]:
        """Chat constrained to `schema`, with one repair attempt on failure."""
        instruction = (
            "Respond with a single JSON object and nothing else -- no prose, no markdown "
            "fences. It must validate against this JSON Schema:\n"
            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
        )
        convo = [*messages, {"role": "system", "content": instruction}]

        res = self.chat(convo, model=model, prompt_version=prompt_version,
                        max_tokens=max_tokens)
        try:
            return schema.model_validate_json(_extract_json(res.text)), res
        except (ValidationError, ValueError, json.JSONDecodeError) as first:
            # Bind inside the block: Python deletes the `except ... as` name on exit.
            why = str(first)[:600]
            log.warning("structured output invalid, attempting one repair: %s",
                        redact(why)[:300])

        repair = [
            *convo,
            {"role": "assistant", "content": res.text[:4000]},
            {"role": "user",
             "content": f"That did not validate: {why}\n"
                        f"Return ONLY the corrected JSON object."},
        ]
        # Not cached: the repair is keyed on a response we do not want to pin.
        res2 = self.chat(repair, model=model, prompt_version=f"{prompt_version}-repair",
                         max_tokens=max_tokens, use_cache=False)
        try:
            return schema.model_validate_json(_extract_json(res2.text)), res2
        except (ValidationError, ValueError, json.JSONDecodeError) as second:
            raise StructuredOutputError(
                "model failed to produce schema-valid output after one repair; "
                "treat the prompt or schema as the defect",
                raw=res2.text,
                context={"schema": schema.__name__, "error": str(second)[:300]},
            ) from second

    # Retry loop

    def _chat_with_retries(self, messages, model, temperature, max_tokens) -> ChatResult:
        last: Exception | None = None

        for attempt in range(1, self.settings.max_attempts + 1):
            started = time.monotonic()
            try:
                # Shed load before spending a socket on a service we believe is down.
                # Inside the try so the CircuitOpenError handler below can decide whether
                # this request's real cause is the breaker or the upstream error.
                self.breaker.before_call()
                res = self._post_chat(messages, model, temperature, max_tokens)
                self.breaker.on_success()
                res.attempts = attempt
                log.info("chat ok", extra={"agenttrace": {
                    "model": model, "attempt": attempt,
                    "latency_ms": int((time.monotonic() - started) * 1000),
                    "cost_inr": round(res.cost_inr, 5),
                    "budget": self.budget.snapshot(),
                }})
                return res

            except (AuthError, BadRequestError, TruncatedResponseError) as terminal:
                # Our credentials or payload, not upstream health. Fails immediately and
                # does not count toward opening the circuit.
                log.error("terminal upstream error, not retrying: %s", terminal)
                raise

            except (RateLimitError, ServerError, TimeoutError_) as transient:
                last = transient
                self.breaker.on_failure()
                if attempt >= self.settings.max_attempts:
                    break
                delay = self._backoff(attempt, transient.retry_after_s)
                log.warning("attempt %d/%d failed (%s), sleeping %.2fs",
                            attempt, self.settings.max_attempts,
                            type(transient).__name__, delay)
                self._sleep(delay)

            except CircuitOpenError:
                # If this request's own retries tripped the breaker, its real cause is the
                # upstream error, so report that and let the next caller be shed. Only a
                # circuit already open before attempt 1 propagates.
                #
                # `from None` replaces rather than chains, keeping the upstream error at
                # the top of the traceback.
                if last is not None:
                    raise last from None
                raise

        if last is not None:
            raise last
        # Unreachable unless the loop's control flow changes; a bug, not an upstream fault.
        raise ServerError("retry loop exited with no recorded error -- this is a bug")

    def _backoff(self, attempt: int, retry_after_s: float | None) -> float:
        """Full jitter, with the server's Retry-After as a floor."""
        cap = min(self.settings.backoff_max_s,
                  self.settings.backoff_base_s * (2 ** (attempt - 1)))
        jittered = random.uniform(0.0, cap)
        if retry_after_s is not None:
            # Never sooner than the server asked; jitter on top to de-synchronise.
            return retry_after_s + random.uniform(0.0, self.settings.backoff_base_s)
        return jittered

    def _post_chat(self, messages, model, temperature, max_tokens) -> ChatResult:
        payload = {"model": model, "messages": messages,
                   "temperature": temperature, "max_tokens": max_tokens}
        headers = {"api-subscription-key": self.settings.require_key(),
                   "Content-Type": "application/json"}
        try:
            r = self._client.post("/v1/chat/completions", json=payload, headers=headers)
        except httpx.TimeoutException as exc:
            raise TimeoutError_(f"request exceeded {self.settings.request_timeout_s}s",
                                context={"model": model}) from exc
        except httpx.HTTPError as exc:
            # Connection reset, DNS or TLS failure: transient at this layer.
            raise ServerError(f"transport failure: {type(exc).__name__}",
                              context={"model": model}) from exc

        if r.status_code >= 400:
            raise _classify(r.status_code, r.text, r.headers)

        try:
            body = r.json()
            choice = body["choices"][0]
            message = choice["message"]
            # `content` is present-but-null on a truncated response, so read it explicitly.
            text = message.get("content")
            finish_reason = choice.get("finish_reason")
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            # A 200 with an unexpected shape is a contract change, not a transient fault.
            raise BadRequestError(
                "unexpected chat-completions response shape",
                status=r.status_code,
                context={"body": redact(r.text)[:400]},
            ) from exc

        usage = body.get("usage") or {}
        in_tok = int(usage.get("prompt_tokens", 0) or 0)
        out_tok = int(usage.get("completion_tokens", 0) or 0)
        details = usage.get("prompt_tokens_details") or {}
        cached = int(details.get("cached_tokens", 0) or 0)
        out_details = usage.get("completion_tokens_details") or {}
        reasoning = int(out_details.get("reasoning_tokens", 0) or 0)
        if not reasoning and (rc := message.get("reasoning_content")):
            reasoning = len(rc) // 4          # estimate when the API does not itemise

        # Empty content with finish_reason=length is a token-budget failure. Raising here
        # keeps the error next to its cause rather than surfacing as a schema error later.
        if not (text or "").strip() and finish_reason == "length":
            raise TruncatedResponseError(
                "model spent its entire token budget reasoning and emitted no content",
                status=r.status_code, max_tokens=max_tokens,
                finish_reason=finish_reason, reasoning_tokens=reasoning)

        return ChatResult(
            text=text or "",
            reasoning_tokens=reasoning,
            finish_reason=finish_reason,
            input_tokens=max(0, in_tok - cached),
            cached_input_tokens=cached,
            output_tokens=out_tok,
            cost_inr=BudgetGuard.estimate_inr(input_tokens=max(0, in_tok - cached),
                                              output_tokens=out_tok,
                                              cached_tokens=cached),
            model=model,
        )

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict:
        return {"breaker": self.breaker.snapshot(), "budget": self.budget.snapshot(),
                "cache": self.cache.stats()}


def _extract_json(text: str) -> str:
    """Extract the JSON object from a response that may be fenced or prose-wrapped."""
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.split("```")[1] if "```" in s[3:] else s[3:]
        s = s.removeprefix("json").strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found in response: {redact(s)[:200]!r}")
    return s[start:end + 1]
