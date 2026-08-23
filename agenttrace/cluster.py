"""Canonicalise open-set labels into clusters.

Open-set labeling produces synonyms (`disputes_amount`, `amount_dispute`,
`disputed_outstanding_amount`) for one underlying situation. This module merges them.

Matching is string-based rather than embedding-based so a merge is explainable: shared
tokens can be checked by the person questioning the merge, a similarity score cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .costs import Usage, component_cost, platform_cost
from .label import SituationLabel
from .models import Conversation

if TYPE_CHECKING:
    from .domains import DomainPack

# Non-discriminating tokens when comparing situation slugs.
_STOP = {"the", "a", "an", "of", "to", "for", "and", "is", "call", "caller", "customer",
         "borrower", "agent", "request", "requests", "issue", "query", "about"}

# Aliases live in the domain pack: a merge decision is domain knowledge
# ("wants_statement is an amount dispute" holds for collections, not for KYC).


def _aliases(domain: DomainPack | None) -> dict[str, str]:
    from .domains import load_domain
    return (domain or load_domain()).aliases


def normalise(slug: str, domain: DomainPack | None = None) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (slug or "").lower()).strip("_")
    return _aliases(domain).get(s, s)


def _tokens(slug: str, domain: DomainPack | None = None) -> frozenset[str]:
    parts = [p for p in normalise(slug, domain).split("_") if p and p not in _STOP]
    # Crude singularisation so "disputes" and "dispute" share a token.
    return frozenset(p[:-1] if len(p) > 4 and p.endswith("s") else p for p in parts)


# 0.5 merges observed same-gap pairs such as auto_debit_bounce_query /
# auto_debit_failure_query. Safe at this level because of the `protected` guard below.
SIMILARITY_THRESHOLD = 0.5


def _similar(a: str, b: str, *, threshold: float = SIMILARITY_THRESHOLD,
             domain: DomainPack | None = None,
             protected: frozenset[str] = frozenset()) -> bool:
    """Whether two situation slugs denote the same cluster.

    `protected` holds slugs the suite declares separately; those never merge into each
    other regardless of name similarity, so each keeps its own coverage verdict.
    """
    if a in protected and b in protected:
        return False
    ta, tb = _tokens(a, domain), _tokens(b, domain)
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= threshold


@dataclass
class Cluster:
    """A situation occurring in production, with the fields needed to prioritise it."""

    key: str
    label: str
    conversation_ids: list[str] = field(default_factory=list)
    member_slugs: set[str] = field(default_factory=set)   # raw labels merged in here
    failures: int = 0
    compliance_hits: int = 0
    conditions: dict[str, int] = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    cost_inr: float = 0.0
    usage: Usage = field(default_factory=Usage)
    by_version: dict[str, dict] = field(default_factory=dict)
    flagged_new: int = 0
    confidence_sum: float = 0.0

    @property
    def volume(self) -> int:
        return len(self.conversation_ids)

    @property
    def fail_rate(self) -> float:
        return self.failures / self.volume if self.volume else 0.0

    @property
    def compliance_rate(self) -> float:
        return self.compliance_hits / self.volume if self.volume else 0.0

    @property
    def mean_confidence(self) -> float:
        return self.confidence_sum / self.volume if self.volume else 0.0

    @property
    def failed_cost_inr(self) -> float:
        """Rupees spent on the calls in this cluster that failed."""
        return self.cost_inr * self.fail_rate

    def top_conditions(self, n: int = 4) -> list[tuple[str, int]]:
        return sorted(self.conditions.items(), key=lambda kv: -kv[1])[:n]

    def exemplars(self, n: int = 3) -> list[str]:
        return self.conversation_ids[:n]


def build_clusters(
    conversations: list[Conversation],
    labels: dict[str, SituationLabel],
    *,
    cost_mode: str = "component",
    min_cluster_size: int = 3,
    domain: DomainPack | None = None,
    protected: frozenset[str] = frozenset(),
) -> tuple[list[Cluster], list[Cluster]]:
    """Group labelled conversations into clusters.

    Returns (clusters, tail). Clusters below `min_cluster_size` are returned separately so
    the long tail is reported rather than silently dropped from the totals.
    """
    by_conv = {c.id: c for c in conversations}
    canonical: dict[str, str] = {}          # raw normalised slug -> cluster key
    buckets: dict[str, Cluster] = {}

    for conv_id, label in labels.items():
        conv = by_conv.get(conv_id)
        if conv is None:
            continue
        raw = normalise(label.situation, domain)

        if raw in canonical:
            key = canonical[raw]
        else:
            # First-match rather than best-match: keeps the merge easy to explain.
            key = next((k for k in buckets
                        if _similar(raw, k, domain=domain, protected=protected)), raw)
            canonical[raw] = key

        cl = buckets.setdefault(key, Cluster(key=key,
                                             label=label.situation_label or key.replace("_", " ")))
        cl.member_slugs.add(raw)
        cl.conversation_ids.append(conv_id)
        cl.confidence_sum += label.confidence
        if label.is_new_situation:
            cl.flagged_new += 1
        if label.agent_failed:
            cl.failures += 1
            # De-duplicated: the same failure mode repeated three times reads as three
            # findings when it is one, and pads the UI with nothing.
            if (label.failure_mode and len(cl.failure_modes) < 6
                    and label.failure_mode not in cl.failure_modes):
                cl.failure_modes.append(label.failure_mode)
        if label.compliance_flags:
            cl.compliance_hits += 1
            for f in label.compliance_flags:
                cl.conditions[f"compliance:{f}"] = cl.conditions.get(f"compliance:{f}", 0) + 1
        for cond in label.conditions:
            cl.conditions[cond] = cl.conditions.get(cond, 0) + 1

        usage = conv.estimated_usage()
        cl.usage = cl.usage + usage
        breakdown = (platform_cost(usage) if cost_mode == "platform"
                     else component_cost(usage))
        cl.cost_inr += breakdown.total

        v = cl.by_version.setdefault(conv.agent_version, {"n": 0, "failed": 0, "cost_inr": 0.0})
        v["n"] += 1
        v["failed"] += int(label.agent_failed)
        v["cost_inr"] += breakdown.total

    all_clusters = sorted(buckets.values(), key=lambda c: -c.volume)
    main = [c for c in all_clusters if c.volume >= min_cluster_size]
    tail = [c for c in all_clusters if c.volume < min_cluster_size]
    return main, tail
