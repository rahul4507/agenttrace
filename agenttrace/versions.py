"""Version attribution: localising a regression to a cluster.

Conversations carry `agent_version` and clusters carry per-version counts, so comparing two
versions per cluster identifies which situation moved rather than reporting a single global
metric.

The difficulty is avoiding false positives. A cluster going from 2 failures in 18 to 5 in
18 is a 167% relative move and consistent with noise; alerting on it trains people to
ignore the alert. Every comparison therefore carries a significance test, and comparisons
below the sample floor return INSUFFICIENT_DATA rather than a verdict.

Fisher's exact test rather than a normal approximation: per-cluster tables are small, which
is where the approximation is least reliable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from .cluster import Cluster
from .coverage import CoverageReport


class Verdict(StrEnum):
    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    UNCHANGED = "unchanged"
    INSUFFICIENT_DATA = "insufficient_data"
    NEW_CLUSTER = "new_cluster"          # appears only in the candidate version
    DISAPPEARED = "disappeared"          # present in baseline, absent in candidate


# Below this, no comparison is attempted: at 8 calls a side even a 0-to-3 swing does not
# reach p<0.05 on Fisher's exact test.
MIN_SAMPLES_PER_SIDE = 8
ALPHA = 0.05


def fisher_exact_greater(a: int, b: int, c: int, d: int) -> float:
    """One-sided p-value for a 2x2 table, testing whether the candidate is worse.

        table = [[a, b],   a = baseline failures,  b = baseline passes
                 [c, d]]   c = candidate failures, d = candidate passes

    One-sided because the question is directional; improvements are tested by swapping the
    rows. Computes the hypergeometric tail via math.comb, so no scipy dependency.
    """
    row1, row2 = a + b, c + d
    col1, total = a + c, a + b + c + d
    if total == 0 or row1 == 0 or row2 == 0 or col1 == 0:
        return 1.0

    def prob(x: int) -> float:
        return (math.comb(row1, x) * math.comb(row2, col1 - x)) / math.comb(total, col1)

    # Sum the tail where the candidate has at least as many failures as observed.
    lo = max(0, col1 - row2)
    p = 0.0
    for x in range(lo, min(row1, col1) + 1):
        # x = baseline failures in this arrangement; candidate failures = col1 - x.
        if (col1 - x) >= c:
            p += prob(x)
    return min(1.0, p)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Wald is unusable here: it admits negative rates and collapses to zero width at zero
    failures.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


@dataclass
class ClusterDiff:
    key: str
    label: str
    baseline_n: int
    baseline_failures: int
    candidate_n: int
    candidate_failures: int
    verdict: Verdict
    p_value: float = 1.0
    baseline_ci: tuple[float, float] = (0.0, 1.0)
    candidate_ci: tuple[float, float] = (0.0, 1.0)
    is_critical: bool = False

    @property
    def baseline_rate(self) -> float:
        return self.baseline_failures / self.baseline_n if self.baseline_n else 0.0

    @property
    def candidate_rate(self) -> float:
        return self.candidate_failures / self.candidate_n if self.candidate_n else 0.0

    @property
    def delta(self) -> float:
        return self.candidate_rate - self.baseline_rate

    @property
    def blocking(self) -> bool:
        """Whether this diff should fail a CI gate. Significant regressions only."""
        return self.verdict is Verdict.REGRESSION

    def to_dict(self) -> dict:
        return {
            "key": self.key, "label": self.label, "verdict": str(self.verdict),
            "baseline": {"n": self.baseline_n, "failures": self.baseline_failures,
                         "rate": round(self.baseline_rate, 3),
                         "ci": [round(x, 3) for x in self.baseline_ci]},
            "candidate": {"n": self.candidate_n, "failures": self.candidate_failures,
                          "rate": round(self.candidate_rate, 3),
                          "ci": [round(x, 3) for x in self.candidate_ci]},
            "delta": round(self.delta, 3),
            "p_value": round(self.p_value, 5),
            "blocking": self.blocking,
            "is_critical": self.is_critical,
        }

    def explain(self) -> str:
        if self.verdict is Verdict.INSUFFICIENT_DATA:
            return (f"{self.label}: {self.baseline_n} vs {self.candidate_n} calls -- too "
                    f"few to distinguish signal from noise (need {MIN_SAMPLES_PER_SIDE} "
                    f"per side)")
        if self.verdict is Verdict.NEW_CLUSTER:
            return (f"{self.label}: NEW in the candidate version ({self.candidate_n} calls, "
                    f"{100*self.candidate_rate:.0f}% failing) -- did not occur in baseline")
        if self.verdict is Verdict.DISAPPEARED:
            return f"{self.label}: present in baseline, absent from the candidate"
        if self.verdict is Verdict.UNCHANGED:
            return (f"{self.label}: {100*self.baseline_rate:.0f}% -> "
                    f"{100*self.candidate_rate:.0f}% (p={self.p_value:.2f}, not significant)")
        direction = "REGRESSED" if self.verdict is Verdict.REGRESSION else "improved"
        return (f"{self.label}: {direction} {100*self.baseline_rate:.0f}% -> "
                f"{100*self.candidate_rate:.0f}% failure "
                f"({self.baseline_failures}/{self.baseline_n} -> "
                f"{self.candidate_failures}/{self.candidate_n}, p={self.p_value:.4f})")


@dataclass
class VersionComparison:
    baseline: str
    candidate: str
    diffs: list[ClusterDiff]

    @property
    def regressions(self) -> list[ClusterDiff]:
        return [d for d in self.diffs if d.verdict is Verdict.REGRESSION]

    @property
    def improvements(self) -> list[ClusterDiff]:
        return [d for d in self.diffs if d.verdict is Verdict.IMPROVEMENT]

    @property
    def new_clusters(self) -> list[ClusterDiff]:
        return [d for d in self.diffs if d.verdict is Verdict.NEW_CLUSTER]

    @property
    def should_block(self) -> bool:
        return any(d.blocking for d in self.diffs)

    def overall_rates(self) -> dict:
        """Aggregate failure rate per version.

        Reported next to the per-cluster diffs: an aggregate can move little while one
        cluster moves a lot, and it does not identify which.
        """
        b_n = sum(d.baseline_n for d in self.diffs)
        b_f = sum(d.baseline_failures for d in self.diffs)
        c_n = sum(d.candidate_n for d in self.diffs)
        c_f = sum(d.candidate_failures for d in self.diffs)
        return {
            "baseline": {"n": b_n, "failures": b_f,
                         "rate": round(b_f / b_n, 4) if b_n else 0.0},
            "candidate": {"n": c_n, "failures": c_f,
                          "rate": round(c_f / c_n, 4) if c_n else 0.0},
            "aggregate_delta": round((c_f / c_n if c_n else 0) - (b_f / b_n if b_n else 0), 4),
        }

    def to_dict(self) -> dict:
        return {
            "baseline": self.baseline, "candidate": self.candidate,
            "overall": self.overall_rates(),
            "regressions": [d.to_dict() for d in self.regressions],
            "diffs": [d.to_dict() for d in self.diffs],
            "should_block": self.should_block,
        }


def compare_versions(report: CoverageReport, baseline: str, candidate: str,
                     *, alpha: float = ALPHA) -> VersionComparison:
    diffs: list[ClusterDiff] = []

    for row in report.rows:
        cl: Cluster = row.cluster
        b = cl.by_version.get(baseline)
        c = cl.by_version.get(candidate)

        if b is None and c is None:
            continue
        if b is None:
            diffs.append(ClusterDiff(cl.key, cl.label, 0, 0, c["n"], c["failed"],
                                     Verdict.NEW_CLUSTER, is_critical=row.is_critical,
                                     candidate_ci=wilson_interval(c["failed"], c["n"])))
            continue
        if c is None:
            diffs.append(ClusterDiff(cl.key, cl.label, b["n"], b["failed"], 0, 0,
                                     Verdict.DISAPPEARED, is_critical=row.is_critical,
                                     baseline_ci=wilson_interval(b["failed"], b["n"])))
            continue

        bn, bf, cn, cf = b["n"], b["failed"], c["n"], c["failed"]
        b_ci, c_ci = wilson_interval(bf, bn), wilson_interval(cf, cn)

        if bn < MIN_SAMPLES_PER_SIDE or cn < MIN_SAMPLES_PER_SIDE:
            diffs.append(ClusterDiff(cl.key, cl.label, bn, bf, cn, cf,
                                     Verdict.INSUFFICIENT_DATA, 1.0, b_ci, c_ci,
                                     row.is_critical))
            continue

        # p = P(candidate this bad or worse | no real difference)
        p_worse = fisher_exact_greater(bf, bn - bf, cf, cn - cf)
        # Improvement: same test with rows swapped.
        p_better = fisher_exact_greater(cf, cn - cf, bf, bn - bf)

        if p_worse < alpha and cf / cn > bf / bn:
            verdict, p = Verdict.REGRESSION, p_worse
        elif p_better < alpha and cf / cn < bf / bn:
            verdict, p = Verdict.IMPROVEMENT, p_better
        else:
            verdict, p = Verdict.UNCHANGED, min(p_worse, p_better)

        diffs.append(ClusterDiff(cl.key, cl.label, bn, bf, cn, cf, verdict, p,
                                 b_ci, c_ci, row.is_critical))

    # Regressions first, then by effect size.
    order = {Verdict.REGRESSION: 0, Verdict.NEW_CLUSTER: 1, Verdict.DISAPPEARED: 2,
             Verdict.UNCHANGED: 3, Verdict.IMPROVEMENT: 4, Verdict.INSUFFICIENT_DATA: 5}
    diffs.sort(key=lambda d: (order[d.verdict], -abs(d.delta)))
    return VersionComparison(baseline=baseline, candidate=candidate, diffs=diffs)
