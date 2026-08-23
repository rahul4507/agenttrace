"""Circuit breaker.

Retries handle a transient blip but make a sustained outage worse: N workers times M
attempts against a service that is already down amplifies load and invites a rate-limit
ban. The breaker draws the line between "transient, retry" and "upstream is down, stop".

States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED or OPEN.
"""

from __future__ import annotations

import threading
import time
from enum import StrEnum

from ..errors import CircuitOpenError


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        *,
        fail_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        name: str = "sarvam",
        clock=time.monotonic,
    ) -> None:
        self.fail_threshold = fail_threshold
        self.reset_timeout_s = reset_timeout_s
        self.name = name
        # Injectable clock so the breaker is testable without sleeping.
        self._clock = clock
        self._state = State.CLOSED
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> State:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        """Caller must hold the lock."""
        if self._state is State.OPEN and self._clock() - self._opened_at >= self.reset_timeout_s:
            self._state = State.HALF_OPEN

    def before_call(self) -> None:
        """Raise CircuitOpenError if load is being shed."""
        with self._lock:
            self._maybe_half_open()
            if self._state is State.OPEN:
                reopens_in = self.reset_timeout_s - (self._clock() - self._opened_at)
                raise CircuitOpenError(
                    f"circuit '{self.name}' is open after {self._consecutive_failures} "
                    f"consecutive failures; shedding load",
                    reopens_in_s=max(0.0, reopens_in),
                    context={"state": str(self._state)},
                )
            # HALF_OPEN admits this one call as a probe.

    def on_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._state = State.CLOSED

    def on_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            # A failed probe re-opens immediately rather than waiting for the threshold.
            if self._state is State.HALF_OPEN or self._consecutive_failures >= self.fail_threshold:
                self._state = State.OPEN
                self._opened_at = self._clock()

    def snapshot(self) -> dict:
        with self._lock:
            self._maybe_half_open()
            return {
                "name": self.name,
                "state": str(self._state),
                "consecutive_failures": self._consecutive_failures,
                "fail_threshold": self.fail_threshold,
            }
