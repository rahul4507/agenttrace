"""Pipeline orchestration: transcripts in, coverage report out.

One entry point shared by the CLI, the API and the tests, so there is a single definition
of a run and no path can produce different numbers for the same corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .cluster import build_clusters
from .config import Settings, load_settings
from .coverage import CoverageReport, build_coverage
from .graders import Grade, grade
from .ingest.base import IngestReport
from .ingest.jsonl import JsonlSource
from .label import (
    HeuristicLabeler,
    Labeler,
    LabelRun,
    LlmLabeler,
    label_corpus,
    load_known_slugs,
    save_known_slugs,
)
from .llm.client import SarvamChatClient
from .models import Conversation
from .suite import Suite, load_suite

log = logging.getLogger("agenttrace.report")


@dataclass
class RunArtifacts:
    """Everything a run produced, so no consumer recomputes a number."""

    conversations: list[Conversation]
    ingest: IngestReport
    suite: Suite
    labels: LabelRun
    coverage: CoverageReport
    grades: dict[str, Grade]
    settings: Settings
    domain: str = "collections"

    def labeler_by_conversation(self) -> dict[str, str]:
        """Which labeler produced each label. Surfaced per call, because a report built
        partly on the degraded labeler has to say so at the point of use, not only in a
        summary line."""
        return {r.conversation_id: r.labeler for r in self.labels.results if r.ok}

    def suite_pass_rate(self) -> float:
        """Pass rate over graded conversations only.

        Including ungraded calls would inflate the number with calls no scenario tests.
        """
        if not self.grades:
            return 0.0
        return sum(1 for g in self.grades.values() if g.passed) / len(self.grades)


def run_report(
    *,
    transcripts: Path,
    suite_dir: Path | None = None,
    settings: Settings | None = None,
    use_llm: bool = False,
    cost_mode: str = "component",
    min_cluster_size: int = 3,
    domain: str = "collections",
    converge: bool = False,
    progress=None,
) -> RunArtifacts:
    """Run one report end to end.

    `domain` is threaded to every stage holding domain knowledge: the keyword labeler's
    rules, the cluster alias table and the coverage matcher. Selecting the right suite and
    corpus while leaving the labeler on another domain's rules yields a plausible-looking
    but meaningless report.
    """
    from .domains import load_domain

    settings = settings or load_settings()
    pack = load_domain(domain)
    suite = load_suite(suite_dir or pack.suite_dir())

    conversations, ingest = JsonlSource(transcripts).load()
    log.info("%s", ingest.summary())

    # Always constructed: it is the fallback the LLM path degrades to.
    heuristic = HeuristicLabeler(pack)
    labeler: Labeler = heuristic
    fallback: Labeler | None = None
    client: SarvamChatClient | None = None

    if use_llm and not settings.offline:
        client = SarvamChatClient(settings)
        labeler, fallback = LlmLabeler(client), heuristic

    # --converge shows the labeler canonical slugs from previous runs so it reuses an
    # existing name. Costs a fresh pass: `known` is part of the prompt and the cache key.
    known = set(suite.situations)
    if converge:
        prior = load_known_slugs(settings.cache_dir, domain)
        known |= prior
        if prior:
            log.info("converging against %d canonical slugs from previous runs", len(prior))

    try:
        labels = label_corpus(conversations, labeler=labeler, fallback=fallback,
                              known_situations=known, settings=settings,
                              progress=progress)
    finally:
        if client is not None:
            client.close()

    used = {r.labeler for r in labels.results if r.ok}
    degraded = bool(labels.stopped_early) or (len(used) > 1)

    from .cluster import normalise
    protected = frozenset(normalise(s.situation, pack) for s in suite.scenarios)
    clusters, tail = build_clusters(conversations, labels.labels, cost_mode=cost_mode,
                                    min_cluster_size=min_cluster_size, domain=pack,
                                    protected=protected)
    coverage = build_coverage(clusters, tail, suite,
                              total_conversations=len(conversations),
                              labeler="+".join(sorted(used)) or "none",
                              degraded=degraded, domain=pack)

    # Only grade conversations a declared scenario covers: grading against a scenario
    # never written for the situation produces a meaningless failure.
    scenario_for_situation = {s.situation: s for s in suite.scenarios}
    grades: dict[str, Grade] = {}
    label_map = labels.labels
    for conv in conversations:
        lbl = label_map.get(conv.id)
        if lbl is None:
            continue
        sc = scenario_for_situation.get(lbl.situation)
        if sc is not None:
            grades[conv.id] = grade(conv, sc)

    # Recorded regardless of --converge, so a later converging run has something to use.
    save_known_slugs(settings.cache_dir, domain, {c.key for c in clusters})

    return RunArtifacts(conversations=conversations, ingest=ingest, suite=suite,
                        labels=labels, coverage=coverage, grades=grades, settings=settings,
                        domain=pack.name)
