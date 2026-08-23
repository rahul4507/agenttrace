"""HTTP API and dashboard host.

The report is computed once at startup or on an explicit refresh and held in app state.
Recomputing per request would make every interaction slow and, for the LLM labeler, paid.
More importantly every panel must derive from one run: computing the coverage table and
the version diff in separate requests would let them disagree.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

from .config import REPO_ROOT, Settings, load_settings
from .costs import (
    DEFAULT_HUMAN_AGENT_INR_PER_CALL,
    RATES,
    Usage,
    component_cost,
    platform_cost,
    savings_vs_human,
)
from .errors import AgentTraceError
from .generate import generate_scenario
from .report import RunArtifacts, run_report
from .suite import write_scenario
from .versions import compare_versions

log = logging.getLogger("agenttrace.api")

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="AgentTrace", description="Coverage and regression analysis for voice agents")

# Module-level state: a single-tenant tool holding one run at a time.
_state: dict[str, Any] = {"artifacts": None, "settings": None, "use_llm": False}


def _artifacts() -> RunArtifacts:
    art = _state.get("artifacts")
    if art is None:
        raise HTTPException(503, "no report loaded yet; POST /api/refresh")
    return art


def load(*, transcripts: Path | None = None, suite: Path | None = None,
         settings: Settings | None = None, use_llm: bool = False,
         domain: str = "collections") -> RunArtifacts:
    settings = settings or _state.get("settings") or load_settings()
    art = run_report(
        transcripts=transcripts or REPO_ROOT / "fixtures" / f"{domain}_calls.jsonl",
        suite_dir=suite, settings=settings, use_llm=use_llm, domain=domain)
    _state.update(artifacts=art, settings=settings, use_llm=use_llm,
                  transcripts=transcripts, suite=suite, domain=domain)
    return art


@app.on_event("startup")
def _startup() -> None:
    if _state.get("artifacts") is None:
        try:
            load()
        except AgentTraceError as exc:
            # A failed load must not kill the server; the UI renders the error and can
            # offer a refresh.
            log.error("startup report failed: %s", exc)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/report")
def api_report() -> dict:
    art = _artifacts()
    payload = art.coverage.to_dict()
    payload["meta"] = {
        "ingest": art.ingest.summary(),
        "labels": art.labels.summary(),
        "suite_pass_rate": round(art.suite_pass_rate(), 4),
        "graded": len(art.grades),
        "graded_failed": sum(1 for g in art.grades.values() if not g.passed),
        "graded_critical": sum(1 for g in art.grades.values() if g.has_critical_failure),
        "versions": sorted({c.agent_version for c in art.conversations}),
        "use_llm": _state.get("use_llm", False),
        "domain": art.domain,
    }
    return payload


@app.get("/api/cluster/{key}")
def api_cluster(key: str, limit: int = 4) -> dict:
    art = _artifacts()
    row = next((r for r in art.coverage.rows if r.cluster.key == key), None)
    if row is None:
        raise HTTPException(404, f"no cluster {key!r}")
    by_id = {c.id: c for c in art.conversations}

    # Failing exemplars first.
    ids = row.cluster.conversation_ids
    failing = [i for i in ids if (g := art.grades.get(i)) and not g.passed]
    ordered = (failing + [i for i in ids if i not in set(failing)])[:limit]

    return {
        **row.to_dict(),
        "examples": [
            {
                "id": cid,
                "version": by_id[cid].agent_version,
                "language": by_id[cid].language,
                "duration_s": by_id[cid].duration_s,
                "outcome": str(by_id[cid].outcome) if by_id[cid].outcome else None,
                "source_disposition": by_id[cid].disposition,
                "cost_inr": round(component_cost(by_id[cid].estimated_usage()).total, 3),
                # Redacted: this page is shared, screenshotted and pasted into tickets.
                "transcript": by_id[cid].transcript(redacted=True),
                "tools": [{"name": tc.name, "ok": tc.ok} for tc in by_id[cid].tool_calls],
                "grade": (
                    {"passed": g.passed,
                     "critical": g.has_critical_failure,
                     "failures": [{"type": c.type, "detail": c.detail, "severity": c.severity}
                                  for c in g.failures]}
                    if (g := art.grades.get(cid)) else None),
            }
            for cid in ordered if cid in by_id
        ],
    }


@app.get("/api/diff")
def api_diff(baseline: str = "v2", candidate: str = "v3") -> dict:
    art = _artifacts()
    versions = sorted({c.agent_version for c in art.conversations})
    for v in (baseline, candidate):
        if v not in versions:
            raise HTTPException(400, f"unknown version {v!r}; corpus has {versions}")
    return compare_versions(art.coverage, baseline, candidate).to_dict()


@app.get("/api/costs")
def api_costs() -> dict:
    """Per-conversation cost economics."""
    art = _artifacts()
    convs = art.conversations
    if not convs:
        raise HTTPException(503, "no conversations loaded")

    total = Usage()
    for c in convs:
        total = total + c.estimated_usage()

    n = len(convs)
    comp = component_cost(total)
    plat = platform_cost(total)
    per_call_component = comp.total / n
    per_call_platform = plat.total / n

    return {
        "n_conversations": n,
        "rate_card": {k: {"name": r.name, "unit": str(r.unit), "inr": r.inr, "note": r.note}
                      for k, r in RATES.items()},
        "component": {
            "per_call_inr": round(per_call_component, 3),
            "total_inr": round(comp.total, 2),
            "breakdown": comp.rounded(2),
            "share": {k: round(v, 4) for k, v in comp.share().items()},
        },
        "platform": {
            "per_call_inr": round(per_call_platform, 3),
            "total_inr": round(plat.total, 2),
            "breakdown": plat.rounded(2),
        },
        # Managed vs self-orchestrated ratio.
        "managed_premium_x": round(per_call_platform / per_call_component, 2)
        if per_call_component else None,
        "savings_vs_human": savings_vs_human(per_call_component),
        "human_baseline_inr": DEFAULT_HUMAN_AGENT_INR_PER_CALL,
        "wasted_on_failures_inr": round(
            sum(r.cluster.failed_cost_inr for r in art.coverage.rows), 2),
        "note": "Usage is ESTIMATED from transcripts where the source reports none. "
                "Estimates are labelled so nobody quotes them to a customer as measured.",
    }


@app.post("/api/close-gap/{key}")
def api_close_gap(key: str, write: bool = False) -> dict:
    art = _artifacts()
    row = next((r for r in art.coverage.rows if r.cluster.key == key), None)
    if row is None:
        raise HTTPException(404, f"no cluster {key!r}")

    client = None
    settings = _state.get("settings") or load_settings()
    if _state.get("use_llm") and not settings.offline:
        from .llm.client import SarvamChatClient
        client = SarvamChatClient(settings)
    try:
        sc = generate_scenario(row, {c.id: c for c in art.conversations}, client=client)
    finally:
        if client:
            client.close()

    written = None
    if write:
        written = str(write_scenario(sc, _state.get("suite") or settings.suite_dir))

    return {
        "scenario": sc.model_dump(exclude_none=True),
        "written_to": written,
        "note": "Generated scenarios are proposals. They land in the suite directory as "
                "ordinary files for review -- never auto-merged.",
    }


@app.post("/api/refresh")
def api_refresh(use_llm: bool = False) -> dict:
    art = load(transcripts=_state.get("transcripts"), suite=_state.get("suite"),
               settings=_state.get("settings"), use_llm=use_llm,
               domain=_state.get("domain", "collections"))
    return {"ok": True, "conversations": len(art.conversations),
            "labels": art.labels.summary()}


@app.get("/api/health")
def api_health() -> dict:
    art = _state.get("artifacts")
    return {
        "report_loaded": art is not None,
        "conversations": len(art.conversations) if art else 0,
        "degraded": art.coverage.degraded if art else None,
        "labeler": art.coverage.labeler if art else None,
    }


@app.exception_handler(AgentTraceError)
def _agenttrace_error(request, exc: AgentTraceError) -> JSONResponse:
    """Typed errors become structured responses the UI can act on."""
    return JSONResponse(status_code=400,
                        content={"error": type(exc).__name__, "message": exc.message,
                                 "context": exc.context, "retryable": exc.retryable})
