"""The coverage diff.

Takes what production did (clusters) and what the suite declared (scenarios) and returns
the ranked difference.

Gaps are scored:

    priority = volume_share * failure_rate * consequence

`consequence` promotes compliance exposure, so a small cluster with a regulatory finding
outranks a larger one that is merely unhelpful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from .cluster import Cluster, _similar, normalise
from .compliance import CRITICAL_FLAGS
from .suite import Scenario, Suite


class Status(StrEnum):
    COVERED = "covered"        # a scenario declares this situation
    PARTIAL = "partial"        # a scenario matches, but misses conditions production shows
    UNCOVERED = "uncovered"    # production does this and nothing tests it


_CRITICAL_FLAGS = CRITICAL_FLAGS

# A flag must occur in a meaningful share of a cluster before the cluster is treated as a
# compliance finding. Without a threshold, a handful of bad calls flags an otherwise healthy
# cluster and every row on the report reads as critical. The absolute floor keeps small but
# severe clusters from being dismissed for lack of volume.
CRITICAL_RATE = 0.15
CRITICAL_MIN_COUNT = 2


@dataclass
class CoverageRow:
    cluster: Cluster
    status: Status
    matched: list[Scenario] = field(default_factory=list)
    match_reason: str = ""
    missing_conditions: list[str] = field(default_factory=list)
    priority: float = 0.0

    def _critical_counts(self) -> dict[str, int]:
        """Flags clearing both the rate and the absolute floor."""
        vol = self.cluster.volume or 1
        return {c: n for c, n in self.cluster.conditions.items()
                if c in _CRITICAL_FLAGS
                and n >= CRITICAL_MIN_COUNT and n / vol >= CRITICAL_RATE}

    @property
    def is_critical(self) -> bool:
        return bool(self._critical_counts())

    @property
    def compliance_flags(self) -> list[str]:
        """Flags above threshold only."""
        return sorted(c.removeprefix("compliance:") for c in self._critical_counts())

    @property
    def compliance_detail(self) -> list[dict]:
        """Flag with its count and rate."""
        vol = self.cluster.volume or 1
        return [{"flag": c.removeprefix("compliance:"), "count": n,
                 "rate": round(n / vol, 3)}
                for c, n in sorted(self._critical_counts().items(), key=lambda kv: -kv[1])]

    @property
    def suppressed_flags(self) -> list[dict]:
        """Flags observed but below threshold. Reported, not discarded."""
        vol = self.cluster.volume or 1
        return [{"flag": c.removeprefix("compliance:"), "count": n,
                 "rate": round(n / vol, 3)}
                for c, n in sorted(self.cluster.conditions.items(), key=lambda kv: -kv[1])
                if c in _CRITICAL_FLAGS and c not in self._critical_counts()]

    def to_dict(self) -> dict:
        cl = self.cluster
        return {
            "key": cl.key, "label": cl.label, "status": str(self.status),
            "volume": cl.volume, "fail_rate": round(cl.fail_rate, 3),
            "compliance_rate": round(cl.compliance_rate, 3),
            "cost_inr": round(cl.cost_inr, 2),
            "failed_cost_inr": round(cl.failed_cost_inr, 2),
            "priority": round(self.priority, 4),
            "is_critical": self.is_critical,
            "compliance_flags": self.compliance_flags,
            "compliance_detail": self.compliance_detail,
            "suppressed_flags": self.suppressed_flags,
            "matched_scenarios": [s.id for s in self.matched],
            "match_reason": self.match_reason,
            "missing_conditions": self.missing_conditions,
            "top_conditions": cl.top_conditions(),
            "failure_modes": cl.failure_modes[:3],
            "exemplars": cl.exemplars(),
            "member_slugs": sorted(cl.member_slugs),
            "mean_confidence": round(cl.mean_confidence, 2),
            "by_version": {k: {**v, "fail_rate": round(v["failed"] / v["n"], 3) if v["n"] else 0.0}
                           for k, v in sorted(cl.by_version.items())},
        }


@dataclass
class CoverageReport:
    rows: list[CoverageRow]
    tail: list[Cluster]
    suite_size: int
    total_conversations: int
    labeler: str = ""
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    # Headline numbers

    @property
    def uncovered(self) -> list[CoverageRow]:
        return [r for r in self.rows if r.status is Status.UNCOVERED]

    @property
    def partial(self) -> list[CoverageRow]:
        return [r for r in self.rows if r.status is Status.PARTIAL]

    @property
    def covered(self) -> list[CoverageRow]:
        return [r for r in self.rows if r.status is Status.COVERED]

    @property
    def uncovered_volume(self) -> int:
        return sum(r.cluster.volume for r in self.uncovered)

    @property
    def _clustered_volume(self) -> int:
        return sum(r.cluster.volume for r in self.rows)

    @property
    def declared_pct(self) -> float:
        """Share of traffic in a situation the suite declares (covered or partial).

        Reported alongside `coverage_pct` because they are different facts: a suite can
        name most of production and still not test the conditions production shows.

        Traffic-weighted rather than cluster-weighted; cluster-share overstates coverage
        since a few clusters can be most of the calls.
        """
        if not self._clustered_volume:
            return 0.0
        vol = sum(r.cluster.volume for r in self.covered + self.partial)
        return 100.0 * vol / self._clustered_volume

    @property
    def coverage_pct(self) -> float:
        """Share of traffic declared and with no untested conditions.

        The gap to `declared_pct` is work on existing scenarios; traffic outside both needs
        a new scenario.
        """
        if not self._clustered_volume:
            return 0.0
        return 100.0 * sum(r.cluster.volume for r in self.covered) / self._clustered_volume

    @property
    def undeclared_pct(self) -> float:
        """Share of traffic in a situation no scenario declares."""
        if not self._clustered_volume:
            return 0.0
        return 100.0 * sum(r.cluster.volume for r in self.uncovered) / self._clustered_volume

    @property
    def unaddressed_spend_inr(self) -> float:
        return sum(r.cluster.failed_cost_inr for r in self.uncovered + self.partial)

    def ranked_gaps(self, n: int | None = None) -> list[CoverageRow]:
        gaps = sorted(self.uncovered + self.partial, key=lambda r: -r.priority)
        return gaps[:n] if n else gaps

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_conversations": self.total_conversations,
                "suite_size": self.suite_size,
                "clusters": len(self.rows),
                "covered": len(self.covered),
                "partial": len(self.partial),
                "uncovered": len(self.uncovered),
                "coverage_pct": round(self.coverage_pct, 1),
                "declared_pct": round(self.declared_pct, 1),
                "undeclared_pct": round(self.undeclared_pct, 1),
                "partial_volume": sum(r.cluster.volume for r in self.partial),
                "uncovered_volume": self.uncovered_volume,
                "unaddressed_spend_inr": round(self.unaddressed_spend_inr, 2),
                "tail_clusters": len(self.tail),
                "tail_volume": sum(c.volume for c in self.tail),
                "labeler": self.labeler,
                "degraded": self.degraded,
                "notes": self.notes,
            },
            "rows": [r.to_dict() for r in self.rows],
        }


def _consequence(cluster: Cluster) -> float:
    """Multiplier promoting regulatory exposure above ordinary quality problems.

    Shares CoverageRow.is_critical's threshold so ranking and labelling agree.
    """
    vol = cluster.volume or 1
    if any(n >= CRITICAL_MIN_COUNT and n / vol >= CRITICAL_RATE
           for c, n in cluster.conditions.items() if c in _CRITICAL_FLAGS):
        return 3.0
    if cluster.compliance_rate > 0.1:
        return 2.0
    return 1.0


def build_coverage(
    clusters: list[Cluster],
    tail: list[Cluster],
    suite: Suite,
    *,
    total_conversations: int,
    labeler: str = "",
    degraded: bool = False,
    domain=None,
) -> CoverageReport:
    declared = {normalise(s.situation, domain): s for s in suite.scenarios}
    total_volume = sum(c.volume for c in clusters) or 1
    rows: list[CoverageRow] = []

    for cl in clusters:
        matched: list[Scenario] = []
        reason = ""

        if cl.key in declared:
            matched = suite.by_situation(declared[cl.key].situation)
            reason = f"exact situation match on {cl.key!r}"
        else:
            for slug, sc in declared.items():
                if _similar(cl.key, slug, domain=domain):
                    matched = suite.by_situation(sc.situation)
                    reason = (f"token overlap between {cl.key!r} and declared {slug!r}: "
                              f"shared {sorted(set(cl.key.split('_')) & set(slug.split('_')))}")
                    break

        if not matched:
            status, missing = Status.UNCOVERED, []
        else:
            # A scenario can name the right situation without testing what production
            # shows. Complicating conditions present in a meaningful share of the cluster
            # and undeclared by any matched scenario make coverage PARTIAL.
            declared_conditions = {c.lower() for s in matched for c in s.conditions}
            observed = {c for c, n in cl.conditions.items()
                        if not c.startswith("compliance:") and n / cl.volume >= 0.25}
            missing = sorted(observed - declared_conditions)
            # Same threshold as is_critical, so one bad call cannot demote a cluster.
            has_compliance = any(
                n >= CRITICAL_MIN_COUNT and n / cl.volume >= CRITICAL_RATE
                for c, n in cl.conditions.items() if c in _CRITICAL_FLAGS)
            status = Status.PARTIAL if (missing or has_compliance) else Status.COVERED

        row = CoverageRow(cluster=cl, status=status, matched=matched,
                          match_reason=reason, missing_conditions=missing)
        volume_share = cl.volume / total_volume
        row.priority = volume_share * cl.fail_rate * _consequence(cl)
        # A covered cluster is not a gap.
        if status is Status.COVERED:
            row.priority = 0.0
        rows.append(row)

    rows.sort(key=lambda r: -r.priority)
    notes = []
    if tail:
        notes.append(
            f"{len(tail)} clusters below the minimum size ({sum(c.volume for c in tail)} "
            f"calls) are excluded from ranking and shown as the long tail -- not dropped.")
    if degraded:
        notes.append(
            "Some labels came from the heuristic fallback because the model was "
            "unavailable. Treat cluster boundaries in this report as provisional.")
    return CoverageReport(rows=rows, tail=tail, suite_size=len(suite),
                         total_conversations=total_conversations,
                         labeler=labeler, degraded=degraded, notes=notes)
