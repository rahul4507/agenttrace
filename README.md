# AgentTrace

Coverage and regression analysis for voice agents. Given a set of production call
transcripts and a declared scenario suite, it reports which situations production is
producing that the suite does not test, ranks them, and attributes quality regressions to
individual situations rather than to a global metric.

Built against Sarvam's Voice Agents platform, but the ingest layer is pluggable.

## What it does

```
transcripts ──▶ situation labeling ──▶ clustering ──▶ coverage diff ──▶ ranked gaps
  (adapters)        (open-set)         (canonical)     (vs suite)     (vol × fail × consequence)
                                                            │
                                         ┌──────────────────┴──────────────────┐
                                         ▼                                     ▼
                              scenario generation                    version attribution
```

1. **Ingest.** Adapters normalise every source to one `Conversation` type: the Sarvam
   call-log API, a JSONL export, a CSV. Adding a source does not change the analysis.
2. **Label.** Sarvam-105B with structured output. The labeler is *open-set*: it receives the
   known taxonomy and explicit permission to name a situation outside it. A closed-set
   classifier cannot surface a coverage gap, because it maps every unfamiliar call onto a
   known label and reports full coverage.
3. **Cluster.** Canonicalise the synonyms open-set labeling produces (`amount_dispute` →
   `disputes_amount`). String-based rather than embedding-based, so a merge is explainable
   by the tokens two slugs share.
4. **Diff.** Match clusters against declared scenarios: `covered`, `partial`, `uncovered`.
   `partial` matters — a scenario can name the right situation without testing the
   conditions production actually shows.
5. **Rank.** `volume_share × failure_rate × consequence`, where consequence promotes
   regulatory exposure, so a small cluster with a compliance finding outranks a larger one
   that is merely unhelpful.
6. **Generate.** Turn a gap into a scenario YAML. Assertions derive from compliance flags
   and observed conditions, and the file lands in the suite directory for review.
7. **Attribute.** Per-cluster version comparison with a one-sided Fisher exact test, so a
   regression is localised to a situation.

## Quickstart

```bash
make venv         # .venv + dependencies (Python 3.12+)
make test         # 80 tests, ~1s, no network

make report       # coverage report
make diff         # v2 -> v3 per cluster, with significance tests
make gap          # generate scenarios for the top uncovered clusters
make gate         # exits 1 on a coverage or compliance regression
make kyc          # the same pipeline on a second vertical
make serve        # dashboard on http://127.0.0.1:8078

cp .env.example .env   # add SARVAM_API_KEY for the two commands that need it
make report-llm   # label with Sarvam-105B (~Rs 0.08/conversation, cached afterwards)
make agreement    # inter-labeler agreement
```

Everything except `report-llm` and `agreement` runs with no network and no API key. CI uses
only the offline path.

## Two coverage numbers

`declared` is the share of traffic in a situation the suite names at all. `coverage` (fully
covered) additionally requires the conditions production shows to be tested. They are
reported separately because a suite can name most of production while testing few of its
edge conditions, and one combined number obscures which of the two problems you have.

Both are traffic-weighted rather than cluster-weighted; cluster-share overstates coverage,
since a handful of clusters is usually most of the calls.

## Results on the bundled corpus

620 synthetic NBFC collections calls, two agent versions, an 8-scenario suite:

- **62% declared, 14% fully covered.** 228 calls are in situations no scenario names.
- The highest-ranked gaps are compliance findings: debt disclosed to unverified third
  parties, account details released without identity verification, collections continued
  against a deceased borrower.
- **Version attribution.** The aggregate failure rate moves 9 points between v2 and v3
  without indicating where. Per-cluster, one situation moves 35 points (7% → 42%,
  p = 0.002) and the other nineteen do not move significantly.

## Label quality

`agenttrace agreement` runs both labelers over the same corpus and reports inter-rater
agreement, using Cohen's kappa so chance agreement is corrected for.

On the bundled corpus: **91% agreement on situation**, so the coverage numbers are largely
independent of the model. **Cohen's kappa 0.355 on agent-failure attribution**, which is
"fair" — meaning which calls count as agent failures does depend materially on the labeler.
Coverage figures are usable; per-cluster failure rates need a human-labelled gold set before
they should be quoted precisely.

## Domain packs

Everything customer-specific lives in `domains/<name>.yaml` (taxonomy, dialogue templates,
keyword rules, cluster aliases) plus a scenario suite in `suite/<name>/`. Two verticals ship:
NBFC collections and bank KYC onboarding, running on identical analysis code.

Compliance mappings live in `compliance.py` rather than in a pack, since they encode
regulatory requirements common to Indian lenders: RBI recovery-agent conduct and KYC Master
Direction, the DPDP Act 2023, and PMLA.

## Reliability

The labeler is an unattended loop of N model calls over a corpus, so:

| Concern | Mechanism |
|---|---|
| Which failures are worth retrying | typed exceptions carrying `retryable` |
| Retry storms | bounded attempts, exponential backoff with full jitter, `Retry-After` as a floor |
| Amplifying an outage | circuit breaker (CLOSED → OPEN → HALF_OPEN) |
| Unbounded spend | rupee budget checked before each call, recorded after |
| Reproducibility | content-addressed cache keyed on prompt version and model |
| Resumability | completed calls persist, so a tripped budget resumes from checkpoint |
| Model unavailable | degradation to the heuristic labeler, declared in the report |
| Trusting the labels | inter-labeler agreement measured with Cohen's kappa |
| PII in logs and caches | redaction at every boundary, India-specific patterns |

Sarvam-105B is a reasoning model: it emits `reasoning_content` before `content`, and
reasoning tokens bill as output at ₹73.2/1M. A small `max_tokens` therefore yields a billed
response with no answer, which the client raises as `TruncatedResponseError` and escalates
once. Reasoning is the dominant cost of a labeling run at roughly ₹0.08/conversation.

## Cost

`costs.py` holds the verified rate card in one table, each rate carrying its billing unit.
Per-conversation cost is reported by component, alongside managed-platform versus
self-orchestrated pricing for the same usage, and rupees spent on failing calls.

## Limitations

- The bundled corpus is **synthetic**. Real borrower transcripts cannot leave a customer's
  environment, so `scripts/gen_corpus.py` generates a corpus shaped like a real book. The
  call-log adapter sits behind the same interface.
- The heuristic labeler's situation accuracy on that corpus is not a meaningful measure: the
  corpus templates and the keyword rules share an author. Use `agenttrace agreement`.
- **Failure attribution is the weak link** (kappa 0.355). Coverage is not.
- Open-set labeling fragments. Over 620 calls the model produced seven variants of "the
  borrower has died". The alias table absorbs known variants; `--converge` feeds canonical
  slugs back to the labeler, at the cost of a fresh labeling pass since `known` is part of
  the prompt and therefore the cache key.
- No multiple-comparison correction across per-cluster significance tests. Benjamini–Hochberg
  would be the right addition; family-wise correction would cost too much power.
- `agenttrace/api.py` holds one report in module state, with no auth or multi-tenancy.
- `sarvam_calllogs.py` is written against an unverified reading of the call-log shape. Field
  mappings live in one dict, and a high reject rate is logged rather than hidden.

## Layout

```
agenttrace/          pipeline: ingest, label, cluster, coverage, versions, generate
  llm/               client, budget, circuit breaker, cache
  ingest/            adapters
  web/index.html     dashboard
domains/             domain packs
suite/<domain>/      declared scenarios
fixtures/            generated corpora
scripts/             corpus generator
tests/               80 tests
```
