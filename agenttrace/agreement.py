"""Inter-labeler agreement.

Estimates how much of a coverage report depends on the model, by running two labelers with
independent failure modes over the same corpus:

  the keyword labeler, deterministic and offline
  Sarvam-105B, open-set and reasoning over the full transcript

Agreement indicates a label is probably right; disagreement localises which judgements are
model-dependent.

Cohen's kappa rather than raw agreement. On a corpus that is ~70% non-failures, two
labelers that both answered "not failed" every time would score 70% raw agreement. Kappa
corrects for chance. Landis and Koch's reading: >0.80 almost perfect, 0.61-0.80
substantial, 0.41-0.60 moderate, 0.21-0.40 fair, <=0.20 poor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .cluster import normalise
from .label import SituationLabel


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters over the same items.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for worse than chance.
    """
    if not a or len(a) != len(b):
        raise ValueError("raters must be non-empty and the same length")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    # Expected agreement if each rater kept its own marginal rate but chose independently.
    pa, pb = sum(a) / n, sum(b) / n
    expected = pa * pb + (1 - pa) * (1 - pb)

    if expected >= 1.0:            # both raters constant and identical
        return 1.0
    return (observed - expected) / (1 - expected)


def interpret_kappa(k: float) -> str:
    if k > 0.80:
        return "almost perfect"
    if k > 0.60:
        return "substantial"
    if k > 0.40:
        return "moderate"
    if k > 0.20:
        return "fair"
    return "poor"


@dataclass
class Disagreement:
    conversation_id: str
    a_situation: str
    b_situation: str
    a_failed: bool
    b_failed: bool

    @property
    def kind(self) -> str:
        same_sit = normalise(self.a_situation) == normalise(self.b_situation)
        if same_sit:
            return "failure_only"
        if self.a_failed == self.b_failed:
            return "situation_only"
        return "both"


@dataclass
class AgreementReport:
    a_name: str
    b_name: str
    n: int
    situation_agreement: float
    failure_agreement: float
    failure_kappa: float
    a_failure_rate: float
    b_failure_rate: float
    a_new_rate: float
    b_new_rate: float
    disagreements: list[Disagreement] = field(default_factory=list)
    a_only_situations: set[str] = field(default_factory=set)
    b_only_situations: set[str] = field(default_factory=set)

    @property
    def kappa_reading(self) -> str:
        return interpret_kappa(self.failure_kappa)

    def to_dict(self) -> dict:
        from collections import Counter
        kinds = Counter(d.kind for d in self.disagreements)
        return {
            "labelers": [self.a_name, self.b_name],
            "n": self.n,
            "situation_agreement": round(self.situation_agreement, 4),
            "failure_agreement": round(self.failure_agreement, 4),
            "failure_kappa": round(self.failure_kappa, 4),
            "kappa_reading": self.kappa_reading,
            "failure_rate": {self.a_name: round(self.a_failure_rate, 4),
                             self.b_name: round(self.b_failure_rate, 4)},
            "new_situation_rate": {self.a_name: round(self.a_new_rate, 4),
                                   self.b_name: round(self.b_new_rate, 4)},
            "disagreements": {"total": len(self.disagreements), **dict(kinds)},
            "situations_only_in": {self.a_name: sorted(self.a_only_situations),
                                  self.b_name: sorted(self.b_only_situations)},
        }


def compare_labels(
    a: dict[str, SituationLabel],
    b: dict[str, SituationLabel],
    *,
    a_name: str = "a",
    b_name: str = "b",
) -> AgreementReport:
    """Compare two labelers over the conversations both labelled.

    Restricted to the intersection, so a missing label is not scored as disagreement.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        raise ValueError("the two labelers share no conversations")

    a_sit = [normalise(a[c].situation) for c in shared]
    b_sit = [normalise(b[c].situation) for c in shared]
    a_fail = [bool(a[c].agent_failed) for c in shared]
    b_fail = [bool(b[c].agent_failed) for c in shared]

    disagreements = [
        Disagreement(c, a[c].situation, b[c].situation, a[c].agent_failed, b[c].agent_failed)
        for i, c in enumerate(shared)
        if a_sit[i] != b_sit[i] or a_fail[i] != b_fail[i]
    ]

    n = len(shared)
    return AgreementReport(
        a_name=a_name, b_name=b_name, n=n,
        situation_agreement=sum(1 for x, y in zip(a_sit, b_sit, strict=True) if x == y) / n,
        failure_agreement=sum(1 for x, y in zip(a_fail, b_fail, strict=True) if x == y) / n,
        failure_kappa=cohens_kappa(a_fail, b_fail),
        a_failure_rate=sum(a_fail) / n,
        b_failure_rate=sum(b_fail) / n,
        a_new_rate=sum(1 for c in shared if a[c].is_new_situation) / n,
        b_new_rate=sum(1 for c in shared if b[c].is_new_situation) / n,
        disagreements=disagreements,
        a_only_situations=set(a_sit) - set(b_sit),
        b_only_situations=set(b_sit) - set(a_sit),
    )
