"""PII redaction for logs, caches and compliance assertions.

Under the DPDP Act 2023 we are a Data Processor for borrower data, so writing a phone
number or PAN into an application log creates an unauthorised copy in a system with wider
access and different retention than the system of record. Redaction therefore happens on
the way into any log, cache or rendered page rather than being cleaned up afterwards.

Patterns are India-specific: generic PII regexes miss PAN, Aadhaar, IFSC and UPI.

Overlapping spans are resolved rather than patterns applied in sequence. Sequential
substitution is incorrect here: a 16-digit card contains an Aadhaar-shaped 12-digit prefix,
and a 10-digit phone is also a valid account number. Candidate spans from all patterns are
collected, overlaps resolved by specificity, and the string rewritten once.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Most-specific first; on overlap the earlier entry wins.
_SPECS: list[tuple[str, re.Pattern[str]]] = [
    # Secrets first.
    ("REDACTED_SECRET", re.compile(
        r"(?i)\b(?:api[-_]?key|subscription[-_]?key|authorization|auth|bearer|token|secret|"
        r"password|passwd|pwd)\b\s*[:=]?\s*(?:bearer\s+|token\s+)?[A-Za-z0-9_\-.:/+=]{6,}")),
    ("REDACTED_SECRET", re.compile(r"\bsk-[A-Za-z0-9_\-]{12,}\b")),

    # Card before Aadhaar: 13-19 digits, Luhn-validated (see _luhn_ok).
    ("CARD", re.compile(r"\b\d(?:[ -]?\d){12,18}\b")),
    # Aadhaar: 12 digits, first digit 2-9, often grouped in 4s.
    ("AADHAAR", re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")),
    # PAN: 5 letters, 4 digits, 1 letter.
    ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b", re.IGNORECASE)),
    # IFSC: 4 letters, literal 0, 6 alphanumerics.
    ("IFSC", re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b", re.IGNORECASE)),
    # UPI VPA: handle@psp, no TLD. Before EMAIL for a precise tag.
    ("UPI", re.compile(r"\b[\w.\-]{2,}@(?:okhdfcbank|okicici|oksbi|okaxis|paytm|ybl|ibl|axl|upi)\b",
                       re.IGNORECASE)),
    ("EMAIL", re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    # Indian mobile, optional +91, tolerating internal spaces/hyphens. Before ACCOUNT,
    # since a 10-digit phone also satisfies the 9-18 digit ACCOUNT rule.
    ("PHONE", re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d(?:[\s-]?\d){8}\b")),
    # Bank account: 9-18 digits. Loosest numeric rule, so it runs last.
    ("ACCOUNT", re.compile(r"\b\d{9,18}\b")),
]


def _luhn_ok(text: str) -> bool:
    """Distinguishes card numbers from long account or reference numbers.

    A card failing Luhn still matches the ACCOUNT rule, so the worst case is a less
    specific tag rather than cleartext.
    """
    d = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total, parity = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    tag: str
    rank: int  # index in _SPECS; lower is more specific


def _spans(text: str) -> list[Span]:
    """All non-overlapping PII spans, most-specific pattern winning any conflict."""
    candidates: list[Span] = []
    for rank, (tag, pat) in enumerate(_SPECS):
        for m in pat.finditer(text):
            if tag == "CARD" and not _luhn_ok(m.group(0)):
                continue
            candidates.append(Span(m.start(), m.end(), tag, rank))

    # Prefer the more specific pattern, then the longer match, then the earlier position.
    candidates.sort(key=lambda s: (s.rank, -(s.end - s.start), s.start))
    accepted: list[Span] = []
    for cand in candidates:
        if any(cand.start < a.end and a.start < cand.end for a in accepted):
            continue
        accepted.append(cand)
    return sorted(accepted, key=lambda s: s.start)


def redact(text: str | None) -> str:
    """Replace PII and secrets with type tags.

    Type tags rather than a fixed mask, so redacted log lines remain diagnosable.
    """
    if not text:
        return ""
    out, cursor = [], 0
    for s in _spans(text):
        out.append(text[cursor:s.start])
        out.append(f"[{s.tag}]")
        cursor = s.end
    out.append(text[cursor:])
    return "".join(out)


def contains_pii(text: str | None) -> list[str]:
    """PII types present, de-duplicated, in order of first appearance.

    Also used as a compliance assertion: an agent reading an account number to an
    unverified caller is a finding.
    """
    if not text:
        return []
    seen: list[str] = []
    for s in _spans(text):
        if s.tag != "REDACTED_SECRET" and s.tag not in seen:
            seen.append(s.tag)
    return seen


_SECRET_KEYS = {
    "api_key", "apikey", "api-subscription-key", "api_subscription_key",
    "authorization", "auth", "password", "passwd", "secret", "token", "bearer",
}


def redact_mapping(d: dict, *, keys_to_drop: tuple[str, ...] = ()) -> dict:
    """Recursively redact a dict for structured logging. Named keys are dropped whole."""
    dropped = _SECRET_KEYS | {k.lower() for k in keys_to_drop}
    out: dict = {}
    for k, v in d.items():
        if str(k).lower() in dropped:
            out[k] = "[REDACTED_SECRET]"
        elif isinstance(v, str):
            out[k] = redact(v)
        elif isinstance(v, dict):
            out[k] = redact_mapping(v, keys_to_drop=keys_to_drop)
        elif isinstance(v, (list, tuple)):
            out[k] = [redact(x) if isinstance(x, str)
                      else redact_mapping(x, keys_to_drop=keys_to_drop) if isinstance(x, dict)
                      else x for x in v]
        else:
            out[k] = v
    return out
