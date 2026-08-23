"""Turn an uncovered cluster into a declared scenario.

Two constraints shape this module.

A generated scenario is a proposal. It is written into the suite directory as an ordinary
file so it lands in review, rather than being merged automatically.

Expectations must encode what should happen, not what does. The cluster being generated
from consists of transcripts where the agent behaved badly, and a model asked to "write a
test for these calls" will readily assert the current, wrong behaviour -- after which the
suite passes and the bug ships. The prompt says so explicitly, and the deterministic path
derives assertions from compliance flags and conditions rather than from agent utterances.
"""

from __future__ import annotations

import hashlib
import logging

from pydantic import BaseModel, Field

from .compliance import CONDITION_EXPECTATIONS, FLAG_EXPECTATIONS
from .coverage import CoverageRow
from .llm.client import SarvamChatClient
from .models import Conversation
from .suite import Expectation, Scenario

log = logging.getLogger("agenttrace.generate")

GEN_PROMPT_VERSION = "gen-v2"

# Flag and condition tables live in compliance.py, shared with coverage.py's ranking.

class GeneratedScenario(BaseModel):
    """Model output for a generated scenario.

    The model writes the human-readable parts and may request assertions from a fixed
    allow-list. Expectations themselves come from the compliance tables, so the suite
    cannot accumulate untestable assertions.
    """

    name: str = Field(description="Short scenario name, max 10 words.")
    description: str = Field(description="2-3 sentences: what the caller brings, and what "
                                         "correct handling looks like.")
    persona_temperament: str = Field(description="One word, e.g. distressed, angry, confused.")
    opening_utterance: str = Field(description="What the caller says first, in the same "
                                               "language mix as the real calls.")
    priority: str = Field(description="P0, P1 or P2. P0 only for compliance exposure.")
    request_expectations: list[str] = Field(
        default_factory=list,
        description="Optional extra assertion types from this list ONLY: outcome_in, "
                    "max_turns, must_escalate, no_repeated_agent_turn, "
                    "must_respond_in_caller_language, no_pii_disclosed.")
    tags: list[str] = Field(default_factory=list)


_ALLOWED_REQUESTS = {
    "outcome_in", "max_turns", "must_escalate", "no_repeated_agent_turn",
    "must_respond_in_caller_language", "no_pii_disclosed",
}

_SYSTEM = """You write test scenarios for an Indian NBFC's loan-collections voice agent.

You will be shown a cluster of REAL production calls that the current agent handled BADLY,
and that no existing test covers.

The single most important rule: describe what the agent SHOULD do, never what it currently
does. The transcripts you are shown are examples of FAILURE. A scenario that asserts the
current behaviour would pass forever while the bug stays in production, which is worse than
having no scenario at all.

Write in the same language mix as the real calls (Hindi / Hinglish / English as observed).
Be concrete about the caller. Priority P0 is reserved for calls with regulatory exposure
under RBI recovery-conduct rules or the DPDP Act."""

_USER = """Uncovered cluster: {key}
Volume: {volume} calls ({fail_rate:.0%} handled badly)
Conditions observed: {conditions}
Compliance flags: {flags}
Observed failure modes:
{modes}

Example transcripts of the agent FAILING this situation:
{examples}

Produce the JSON object for a scenario that would have caught this."""


def _derive_expectations(row: CoverageRow) -> list[Expectation]:
    """Expectations implied by the cluster's compliance flags and conditions.

    Deterministic: the same cluster yields the same assertions, each traceable to a named
    regulatory requirement.
    """
    out: list[Expectation] = []
    seen: set[tuple] = set()

    def add(exps: list[Expectation]) -> None:
        for e in exps:
            sig = (e.type, e.tool, e.pattern, tuple(e.patterns or ()), e.value)
            if sig not in seen:
                seen.add(sig)
                out.append(e)

    for flag in row.compliance_flags:
        add(FLAG_EXPECTATIONS.get(flag, []))
    for cond, _ in row.cluster.top_conditions(6):
        add(CONDITION_EXPECTATIONS.get(cond, []))

    # Every scenario needs at least one outcome assertion.
    if not any(e.type == "outcome_in" for e in out):
        add([Expectation(type="outcome_in",
                         values=["resolved", "partial", "escalated"])])
    return out


def _fallback_scenario(row: CoverageRow) -> Scenario:
    """Build a scenario without a model. Assertions are the same as the LLM path's."""
    cl = row.cluster
    pretty = cl.key.replace("_", " ")
    return Scenario(
        # Stable id: Python's hash() is randomised per process (PYTHONHASHSEED), so using
        # it here would give the same cluster a different scenario id on every run -- and
        # a suite whose ids churn cannot be diffed in review or tracked across versions.
        id=f"GEN-{int(hashlib.sha256(cl.key.encode()).hexdigest()[:6], 16) % 10000:04d}",
        name=f"{pretty.title()} (generated from production)",
        situation=cl.key,
        description=(
            f"Derived from {cl.volume} production calls in the '{cl.key}' cluster, of which "
            f"{100*cl.fail_rate:.0f}% were handled badly and none were covered by a "
            f"declared scenario. Observed failure modes: "
            f"{'; '.join(cl.failure_modes[:2]) or 'see exemplars'}."),
        owner="unassigned",
        priority="P0" if row.is_critical else "P1",
        conditions=[c for c, _ in cl.top_conditions(4) if not c.startswith("compliance:")],
        expectations=_derive_expectations(row),
        tags=["generated", "from_production"] + (["compliance"] if row.is_critical else []),
        generated_from_cluster=cl.key,
    )


def generate_scenario(
    row: CoverageRow,
    conversations: dict[str, Conversation],
    *,
    client: SarvamChatClient | None = None,
) -> Scenario:
    """Propose a scenario for an uncovered cluster.

    Without a client the deterministic path produces the same assertions and a plainer
    description, so gap-closing works offline.
    """
    base = _fallback_scenario(row)
    if client is None:
        return base

    cl = row.cluster
    examples = []
    for cid in cl.exemplars(3):
        conv = conversations.get(cid)
        if conv:
            examples.append(f"--- {cid} ---\n{conv.transcript(redacted=True, max_chars=1200)}")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            key=cl.key, volume=cl.volume, fail_rate=cl.fail_rate,
            conditions=", ".join(c for c, _ in cl.top_conditions(6)) or "none",
            flags=", ".join(row.compliance_flags) or "none",
            modes="\n".join(f"  - {m}" for m in cl.failure_modes[:4]) or "  - not recorded",
            examples="\n\n".join(examples) or "(no exemplars available)")},
    ]

    try:
        gen, _ = client.structured(messages, GeneratedScenario,
                                   prompt_version=GEN_PROMPT_VERSION)
    except Exception as exc:
        # Generation is a convenience; falling back beats failing the request.
        log.warning("scenario generation fell back to the deterministic path: %s", exc)
        return base

    extra = [Expectation(type=t) for t in dict.fromkeys(gen.request_expectations)
             if t in _ALLOWED_REQUESTS
             and t not in {e.type for e in base.expectations}]
    if any(e.type == "max_turns" and e.value is None for e in extra):
        extra = [e if e.type != "max_turns" else Expectation(type="max_turns", value=10)
                 for e in extra]
    if any(e.type == "outcome_in" and not e.values for e in extra):
        extra = [e for e in extra if e.type != "outcome_in"]

    base.name = gen.name or base.name
    base.description = gen.description or base.description
    base.persona = base.persona.model_copy(update={"temperament": gen.persona_temperament})
    base.opening_utterance = gen.opening_utterance
    base.priority = gen.priority if gen.priority in {"P0", "P1", "P2"} else base.priority
    base.expectations = base.expectations + extra
    base.tags = sorted(set(base.tags) | set(gen.tags))
    return base
