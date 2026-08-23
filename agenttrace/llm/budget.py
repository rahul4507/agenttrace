"""Spend guardrail for a labeling run.

A labeling run is N model calls, each retried up to `max_attempts`, so a mistaken retry
predicate makes it an unbounded charge. This is a fuse, not a rate limiter or a quota
service: it checks an estimate before spending, records actual usage after, and trips once.

Estimates are deliberately high. A run that stops early resumes from cache; an overspend
cannot be undone.
"""

from __future__ import annotations

import threading

from ..costs import RATES
from ..errors import BudgetExceededError


class BudgetGuard:
    """Thread-safe rupee ceiling for one run."""

    def __init__(self, limit_inr: float, *, name: str = "run") -> None:
        if limit_inr <= 0:
            raise ValueError("limit_inr must be positive")
        self.limit_inr = limit_inr
        self.name = name
        self._spent = 0.0
        self._calls = 0
        self._lock = threading.Lock()

    # Accounting

    @staticmethod
    def estimate_inr(*, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        return (
            RATES["llm_105b_in"].cost(input_tokens)
            + RATES["llm_105b_cached"].cost(cached_tokens)
            + RATES["llm_105b_out"].cost(output_tokens)
        )

    @property
    def spent_inr(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining_inr(self) -> float:
        with self._lock:
            return max(0.0, self.limit_inr - self._spent)

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def fraction_used(self) -> float:
        with self._lock:
            return self._spent / self.limit_inr if self.limit_inr else 1.0

    # The fuse

    def check(self, estimated_inr: float) -> None:
        """Raise before spending if this call would breach the ceiling."""
        with self._lock:
            projected = self._spent + estimated_inr
            if projected > self.limit_inr:
                raise BudgetExceededError(
                    f"budget guard '{self.name}' would be exceeded by this call; "
                    f"stopping so the run can be resumed from its checkpoint",
                    spent_inr=self._spent,
                    limit_inr=self.limit_inr,
                    context={"estimated_inr": round(estimated_inr, 4),
                             "projected_inr": round(projected, 4),
                             "calls_made": self._calls},
                )

    def record(self, actual_inr: float) -> None:
        """Record actual spend after a call returns. Never raises."""
        with self._lock:
            self._spent += max(0.0, actual_inr)
            self._calls += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "limit_inr": round(self.limit_inr, 4),
                "spent_inr": round(self._spent, 4),
                "remaining_inr": round(max(0.0, self.limit_inr - self._spent), 4),
                "calls": self._calls,
                "fraction_used": round(self._spent / self.limit_inr, 4) if self.limit_inr else 1.0,
            }
