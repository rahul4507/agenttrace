"""Generate a synthetic call corpus from a domain pack.

Real borrower transcripts cannot leave a customer's environment, so the fixtures are
generated. The corpus is shaped to match a real book rather than merely reading plausibly:

  a long tail, with most volume in a few situations and the interesting failures in
  clusters of 5-25 calls
  failure rates varying by situation, since that is what the coverage report ranks on
  a split between situations the suite declares and situations it does not
  a version regression planted in one cluster, so attribution has something to find

Seeded, so the corpus is byte-identical across machines.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Filled from the domain pack in main().
OPENING: str = ""
NAMES: list[str] = []
LANGS: list[str] = []
SITUATIONS: dict[str, dict] = {}
REGRESSION_CLUSTER: str = ""
REGRESSION_FAIL_RATE: float = 0.0

# Taxonomy, dialogue templates, names, languages and the planted regression come from the
# domain pack; this file is only the mechanism.
#
# The RNG call sequence is load-bearing: changing it changes the corpus, which invalidates
# every cached model label.


def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


AGENT_ID = "collections-hi"
PRODUCTS = ["personal_loan", "two_wheeler", "consumer_durable"]


def make_conversation(rng: random.Random, sit_key: str, spec: dict, idx: int,
                      version: str, start: datetime) -> dict:
    name = _pick(rng, NAMES)
    amt = rng.choice([2500, 3400, 4500, 6200, 8750, 11400, 15600])
    acct = f"{rng.randrange(10**11, 10**12)}"
    date = f"{rng.randrange(1, 28)} tarikh"
    fill = {"name": name, "amt": f"{amt:,}", "acct": acct, "date": date}

    fail_rate = spec["fail"]
    if sit_key == REGRESSION_CLUSTER and version == "v3":
        fail_rate = REGRESSION_FAIL_RATE

    failed = rng.random() < fail_rate
    turns: list[dict] = []
    n_exchanges = rng.randrange(2, min(4, len(spec["agent"])) + 1)
    lang = spec.get("language") or _pick(rng, LANGS)

    # Outbound calls open with the agent's greeting, then the caller states their reason,
    # then the agent responds. Emitting the situation-specific reply first would make the
    # situation readable from the agent's turn rather than the caller's.
    turns.append({"role": "agent", "language": lang, "offset_ms": 0,
                  "duration_ms": rng.randrange(3000, 5200),
                  "text": OPENING.format(**fill),
                  "tool_calls": ([{"name": "verify_identity",
                                   "arguments": {"loan_id": f"LN{100000+idx}"},
                                   "ok": not (failed and rng.random() < 0.2),
                                   "latency_ms": rng.randrange(120, 900)}]
                                 if "verify_identity" in (spec.get("tools") or []) else [])})

    remaining_tools = [t for t in (spec.get("tools") or []) if t != "verify_identity"]
    for i in range(n_exchanges):
        c_text = _pick(rng, spec["caller"]).format(**fill)
        a_text = spec["agent"][min(i, len(spec["agent"]) - 1)].format(**fill)
        base = (i + 1) * 14_000
        turns.append({"role": "caller", "text": c_text, "language": lang,
                      "offset_ms": base,
                      "duration_ms": rng.randrange(1200, 4200),
                      "asr_confidence": spec.get("asr_conf",
                                                 round(rng.uniform(0.72, 0.97), 2)),
                      "barge_in": rng.random() < 0.18})
        agent_turn = {"role": "agent", "text": a_text, "language": lang,
                      "offset_ms": base + 6_000,
                      "duration_ms": rng.randrange(2500, 6500)}
        if i < len(remaining_tools):
            agent_turn["tool_calls"] = [{
                "name": remaining_tools[i],
                "arguments": {"loan_id": f"LN{100000+idx}"},
                # Tool failures are a minority cause of agent errors.
                "ok": not (failed and rng.random() < 0.25),
                "latency_ms": rng.randrange(120, 1800),
            }]
        turns.append(agent_turn)

    # Outcome per models.Outcome: caller-side endings are not agent failures.
    if failed:
        outcome = ("compliance_breach" if spec.get("compliance") and rng.random() < 0.6
                   else "agent_error")
    else:
        r = rng.random()
        if sit_key in {"wrong_number"}:
            outcome = "resolved"
        elif sit_key == "asks_for_human":
            outcome = "escalated"
        elif r < 0.62:
            outcome = "resolved"
        elif r < 0.80:
            outcome = "partial"
        elif r < 0.90:
            outcome = "escalated"
        else:
            outcome = "caller_abandoned"

    duration = sum(t.get("duration_ms", 0) for t in turns) / 1000.0 + rng.uniform(8, 40)
    return {
        "id": f"call_{version}_{idx:05d}",
        "agent_id": AGENT_ID,
        "agent_version": version,
        "channel": "telephony",
        "language": lang,
        "started_at": start.isoformat(),
        "duration_s": round(duration, 1),
        "turns": turns,
        "outcome": outcome,
        "disposition": sit_key,          # the source's own free-text label
        "source": "synthetic",
        "metadata": {
            "campaign": rng.choice(["bucket-0-30", "bucket-31-60", "bucket-61-90"]),
            "product": rng.choice(PRODUCTS),
            # Ground truth for measuring labeler accuracy. Never shown to the labeler.
            "_truth_situation": sit_key,
            "_truth_failed": failed,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", default="collections",
                    help="domain pack to generate from (domains/<name>.yaml)")
    ap.add_argument("--n", type=int, default=620)
    ap.add_argument("--seed", type=int, default=20260822)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO))
    from agenttrace.domains import load_domain
    pack = load_domain(args.domain)

    global OPENING, NAMES, LANGS, SITUATIONS, REGRESSION_CLUSTER, REGRESSION_FAIL_RATE
    OPENING = pack.opening
    NAMES = pack.names
    LANGS = pack.languages
    SITUATIONS = pack.situations
    REGRESSION_CLUSTER = pack.regression.get("cluster", "")
    REGRESSION_FAIL_RATE = float(pack.regression.get("fail_rate", 0.0))

    global AGENT_ID
    AGENT_ID = pack.name + "-hi"
    global PRODUCTS
    PRODUCTS = pack.situations and (getattr(pack, "products", None) or PRODUCTS)

    out = args.out or (REPO / "fixtures" / f"{args.domain}_calls.jsonl")
    args.out = out

    rng = random.Random(args.seed)
    keys = list(SITUATIONS)
    weights = [SITUATIONS[k]["weight"] for k in keys]

    # v2 is the incumbent, v3 the recent deploy; most traffic is still on v2.
    rows = []
    t0 = datetime(2026, 8, 4, 10, 0, 0)
    for i in range(args.n):
        version = "v2" if i < int(args.n * 0.62) else "v3"
        sit = rng.choices(keys, weights=weights, k=1)[0]
        start = t0 + timedelta(minutes=rng.randrange(0, 24 * 60 * 16))
        rows.append(make_conversation(rng, sit, SITUATIONS[sit], i, version, start))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report the resulting distribution.
    from collections import Counter
    by_sit = Counter(r["metadata"]["_truth_situation"] for r in rows)
    print(f"wrote {len(rows)} conversations -> {args.out.relative_to(REPO)}")
    print(f"{'situation':<34} {'n':>4} {'fail%':>6}  suite")
    for k, n in by_sit.most_common():
        fails = sum(1 for r in rows
                    if r["metadata"]["_truth_situation"] == k and r["metadata"]["_truth_failed"])
        mark = "yes" if SITUATIONS[k]["in_suite"] else "NO  <-- gap"
        print(f"  {k:<32} {n:>4} {100*fails/n:>5.0f}%  {mark}")
    v2 = [r for r in rows if r["agent_version"] == "v2"]
    v3 = [r for r in rows if r["agent_version"] == "v3"]
    print(f"\nversions: v2={len(v2)} v3={len(v3)}  (regression planted in "
          f"'{REGRESSION_CLUSTER}')")


if __name__ == "__main__":
    main()
