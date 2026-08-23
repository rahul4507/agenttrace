"""Behavioural tests for the reliability stack.

Uses a fake httpx transport and an injected sleep, so tests exercising four retries with
exponential backoff run deterministically and without waiting.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agenttrace.config import Settings
from agenttrace.errors import (
    AuthError,
    BadRequestError,
    BudgetExceededError,
    CircuitOpenError,
    ServerError,
    StructuredOutputError,
)
from agenttrace.llm.budget import BudgetGuard
from agenttrace.llm.cache import ResponseCache
from agenttrace.llm.circuit import CircuitBreaker, State
from agenttrace.llm.client import SarvamChatClient


def _body(text: str, *, in_tok: int = 100, out_tok: int = 20, cached: int = 0) -> dict:
    return {
        "choices": [{"message": {"content": text}}],
        "usage": {"prompt_tokens": in_tok, "completion_tokens": out_tok,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    }


class Scripted(httpx.BaseTransport):
    """Replays a scripted list of responses, recording how many calls were made."""

    def __init__(self, script: list[httpx.Response]) -> None:
        self.script = list(script)
        self.calls = 0

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if not self.script:
            raise AssertionError(f"transport called {self.calls}x more than scripted")
        return self.script.pop(0)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(api_key="test-key", cache_dir=tmp_path / "cache",
                    max_attempts=4, backoff_base_s=0.01, backoff_max_s=0.05,
                    run_budget_inr=10.0)


def _client(settings, script, **kw) -> tuple[SarvamChatClient, Scripted, list[float]]:
    t = Scripted(script)
    slept: list[float] = []
    c = SarvamChatClient(settings, transport=t, sleep=slept.append,
                         cache=ResponseCache(settings.cache_dir / "c.db"), **kw)
    return c, t, slept


MSG = [{"role": "user", "content": "hello"}]


# --- retry policy ------------------------------------------------------------------

def test_retries_500_then_succeeds(settings):
    c, t, slept = _client(settings, [
        httpx.Response(500, text="boom"),
        httpx.Response(503, text="boom"),
        httpx.Response(200, json=_body("ok")),
    ])
    res = c.chat(MSG)
    assert res.text == "ok"
    assert res.attempts == 3
    assert t.calls == 3
    assert len(slept) == 2, "sleeps between attempts, not after the last"


def test_auth_error_is_not_retried(settings):
    """A 401 cannot succeed on retry and consumes rate limit."""
    c, t, _ = _client(settings, [httpx.Response(401, text="bad key")])
    with pytest.raises(AuthError):
        c.chat(MSG)
    assert t.calls == 1


def test_bad_request_is_not_retried(settings):
    """A 400 means the payload is wrong, so a retry sends the same invalid request."""
    c, t, _ = _client(settings, [httpx.Response(400, text="unsupported language")])
    with pytest.raises(BadRequestError):
        c.chat(MSG)
    assert t.calls == 1


def test_retries_are_bounded(settings):
    c, t, _ = _client(settings, [httpx.Response(500, text="boom")] * 4)
    with pytest.raises(ServerError):
        c.chat(MSG)
    assert t.calls == settings.max_attempts


def test_retry_after_header_is_honoured_as_a_floor(settings):
    c, t, slept = _client(settings, [
        httpx.Response(429, text="slow down", headers={"Retry-After": "2.5"}),
        httpx.Response(200, json=_body("ok")),
    ])
    assert c.chat(MSG).text == "ok"
    assert slept[0] >= 2.5, "never retries sooner than the server asked"


def test_backoff_is_jittered_not_deterministic(settings):
    """Plain exponential backoff would synchronise workers onto one retry instant."""
    c, _, _ = _client(settings, [])
    delays = {round(c._backoff(3, None), 9) for _ in range(50)}
    assert len(delays) > 40, "full jitter should produce a spread of delays"
    cap = min(settings.backoff_max_s, settings.backoff_base_s * 2 ** 2)
    assert all(0.0 <= d <= cap for d in delays)


def test_200_with_wrong_shape_is_terminal_not_retried(settings):
    """A changed response shape is a contract change, not a transient fault."""
    c, t, _ = _client(settings, [httpx.Response(200, json={"unexpected": True})])
    with pytest.raises(BadRequestError):
        c.chat(MSG)
    assert t.calls == 1


# --- circuit breaker ---------------------------------------------------------------

def test_circuit_opens_and_sheds_load(settings):
    """A request that trips the breaker reports the upstream cause, not the breaker.

    The open circuit is what the next caller sees.
    """
    clock = [0.0]
    breaker = CircuitBreaker(fail_threshold=2, reset_timeout_s=30.0,
                             clock=lambda: clock[0])
    c, t, _ = _client(settings, [httpx.Response(500, text="boom")] * 2, breaker=breaker)
    with pytest.raises(ServerError):
        c.chat(MSG)
    assert breaker.state is State.OPEN
    # Next call must not reach the transport at all.
    before = t.calls
    with pytest.raises(CircuitOpenError):
        c.chat([{"role": "user", "content": "different"}])
    assert t.calls == before, "open circuit must not spend a socket"


def test_circuit_half_opens_after_timeout_and_closes_on_success(settings):
    clock = [0.0]
    breaker = CircuitBreaker(fail_threshold=1, reset_timeout_s=10.0,
                             clock=lambda: clock[0])
    breaker.on_failure()
    assert breaker.state is State.OPEN
    clock[0] = 11.0
    assert breaker.state is State.HALF_OPEN
    breaker.on_success()
    assert breaker.state is State.CLOSED


def test_failed_probe_reopens_immediately(settings):
    """A failure in HALF_OPEN re-opens without waiting for the threshold again."""
    clock = [0.0]
    breaker = CircuitBreaker(fail_threshold=5, reset_timeout_s=10.0,
                             clock=lambda: clock[0])
    for _ in range(5):
        breaker.on_failure()
    clock[0] = 11.0
    assert breaker.state is State.HALF_OPEN
    breaker.on_failure()
    assert breaker.state is State.OPEN


# --- budget fuse -------------------------------------------------------------------

def test_budget_blocks_before_spending(settings):
    guard = BudgetGuard(0.0001, name="tiny")
    c, t, _ = _client(settings, [httpx.Response(200, json=_body("ok"))], budget=guard)
    with pytest.raises(BudgetExceededError) as exc:
        c.chat(MSG)
    assert t.calls == 0, "must trip before the request is made"
    assert exc.value.limit_inr == 0.0001


def test_budget_accumulates_real_usage(settings):
    guard = BudgetGuard(10.0)
    c, _, _ = _client(settings, [httpx.Response(200, json=_body("ok", in_tok=1_000_000,
                                                               out_tok=1_000_000))],
                      budget=guard)
    c.chat(MSG)
    # 1M input @ 29.28 + 1M output @ 73.2
    assert guard.spent_inr == pytest.approx(29.28 + 73.20, rel=1e-6)


# --- cache -------------------------------------------------------------------------

def test_cache_prevents_a_second_network_call(settings):
    c, t, _ = _client(settings, [httpx.Response(200, json=_body("cached me"))])
    first = c.chat(MSG)
    second = c.chat(MSG)
    assert t.calls == 1
    assert not first.from_cache and second.from_cache
    assert second.text == "cached me"


def test_cache_key_includes_prompt_version(settings):
    """Otherwise a report would mix labels produced by two different prompts."""
    c, t, _ = _client(settings, [httpx.Response(200, json=_body("a")),
                                 httpx.Response(200, json=_body("b"))])
    c.chat(MSG, prompt_version="v1")
    c.chat(MSG, prompt_version="v2")
    assert t.calls == 2, "a prompt change must invalidate the cache"


def test_cached_call_does_not_double_charge_the_budget(settings):
    guard = BudgetGuard(10.0)
    c, _, _ = _client(settings, [httpx.Response(200, json=_body("ok"))], budget=guard)
    c.chat(MSG)
    spent_once = guard.spent_inr
    c.chat(MSG)
    assert guard.spent_inr == spent_once


# --- structured output -------------------------------------------------------------

class Label(BaseModel):
    situation: str
    confidence: float


def test_structured_output_parses_fenced_json(settings):
    c, _, _ = _client(settings, [httpx.Response(200, json=_body(
        '```json\n{"situation": "disputes_amount", "confidence": 0.9}\n```'))])
    label, _ = c.structured(MSG, Label)
    assert label.situation == "disputes_amount"


def test_structured_output_repairs_once(settings):
    c, t, _ = _client(settings, [
        httpx.Response(200, json=_body('{"situation": "x"}')),        # missing confidence
        httpx.Response(200, json=_body('{"situation": "x", "confidence": 0.5}')),
    ])
    label, _ = c.structured(MSG, Label)
    assert label.confidence == 0.5
    assert t.calls == 2


def test_structured_output_gives_up_after_one_repair(settings):
    """Failing twice indicates the prompt or schema is wrong, so retrying is futile."""
    c, t, _ = _client(settings, [
        httpx.Response(200, json=_body("not json at all")),
        httpx.Response(200, json=_body("still not json")),
    ])
    with pytest.raises(StructuredOutputError):
        c.structured(MSG, Label)
    assert t.calls == 2
