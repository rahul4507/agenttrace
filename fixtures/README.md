# Fixtures

**`collections_calls.jsonl`, `kyc_calls.jsonl`** — synthetic call corpora, produced by
`scripts/gen_corpus.py` from the corresponding domain pack. Seeded, so they are
byte-identical on every machine. Regenerate with `make corpus` / `make kyc`.

Real borrower transcripts cannot leave a customer's environment, so the demo corpora are
generated. They are shaped to match a real book: a long tail, failure rates varying by
situation, a split between declared and undeclared situations, and a version regression
planted in one cluster.

**`llm_responses.db`** — recorded Sarvam-105B responses for the collections corpus, so
`make report-llm` and `make agreement` run from a clean checkout with no API key and no
cost. Copied into `.cache/responses.db` on first use if no live cache exists.

Each row is model output keyed by a SHA-256 hash of the request. No prompts and no
transcripts are stored. Entries only hit when the corpus and prompt version are unchanged,
so editing either falls back to live calls.
