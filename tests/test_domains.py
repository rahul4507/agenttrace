"""Tests for the domain-pack design.

Assert that the pipeline runs on every shipped vertical, that packs are internally
consistent, and that compliance knowledge has a single source.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agenttrace.compliance import CRITICAL_FLAGS, FLAG_EXPECTATIONS
from agenttrace.config import REPO_ROOT, load_settings
from agenttrace.domains import available_domains, load_domain
from agenttrace.errors import ConfigError
from agenttrace.graders import _GRADERS
from agenttrace.report import run_report
from agenttrace.suite import load_suite

DOMAINS = available_domains()


def test_more_than_one_domain_ships():
    assert "collections" in DOMAINS and "kyc" in DOMAINS


@pytest.mark.parametrize("name", DOMAINS)
def test_pack_loads_and_validates(name):
    pack = load_domain(name)
    assert pack.situations and pack.rule_order


@pytest.mark.parametrize("name", DOMAINS)
def test_every_rule_order_entry_exists(name):
    """A rule naming an unknown situation would never match and never error."""
    pack = load_domain(name)
    assert not set(pack.rule_order) - set(pack.situations)


@pytest.mark.parametrize("name", DOMAINS)
def test_aliases_never_point_at_themselves(name):
    """A self-alias makes canonicalisation a silent no-op."""
    pack = load_domain(name)
    for src, dst in pack.aliases.items():
        assert src != dst, f"{name}: alias {src!r} -> itself"


@pytest.mark.parametrize("name", DOMAINS)
def test_alias_targets_are_real_situations(name):
    pack = load_domain(name)
    unknown = {d for d in pack.aliases.values() if d not in pack.situations}
    assert not unknown, f"{name}: aliases point at unknown situations {unknown}"


@pytest.mark.parametrize("name", DOMAINS)
def test_declared_compliance_flags_have_assertions(name):
    """A flag with no assertion can be detected but not acted on."""
    pack = load_domain(name)
    declared = {f for s in pack.situations.values() for f in (s.get("compliance") or [])}
    missing = declared - set(FLAG_EXPECTATIONS)
    assert not missing, f"{name}: no assertion mapping for {sorted(missing)}"


@pytest.mark.parametrize("name", DOMAINS)
def test_the_suite_matches_the_packs_expectation(name):
    """Situations marked in_suite must be declared by the suite, and vice versa."""
    pack = load_domain(name)
    suite = load_suite(pack.suite_dir())
    assert pack.in_suite() == suite.situations, (
        f"{name}: pack says {sorted(pack.in_suite())}, suite declares "
        f"{sorted(suite.situations)}")


@pytest.mark.parametrize("name", DOMAINS)
def test_suite_expectations_are_all_gradeable(name):
    suite = load_suite(load_domain(name).suite_dir())
    for sc in suite.scenarios:
        for e in sc.expectations:
            assert e.type in _GRADERS, f"{name}/{sc.id}: {e.type!r} has no grader"


@pytest.mark.parametrize("name", DOMAINS)
def test_pipeline_runs_end_to_end_on_every_domain(name):
    """Every vertical runs through the same ingest, labeling, clustering and ranking code."""
    art = run_report(transcripts=REPO_ROOT / "fixtures" / f"{name}_calls.jsonl",
                     domain=name, settings=load_settings(offline=True))
    cov = art.coverage
    assert art.domain == name
    assert cov.total_conversations > 100
    assert cov.rows, "produced no clusters"
    assert cov.uncovered, "a real corpus always has situations the suite missed"
    assert 0.0 < cov.coverage_pct < 100.0


def test_the_two_domains_do_not_bleed_into_each_other():
    """A domain must not be labelled with another domain's rules.

    Selecting the right suite and corpus while leaving the labeler on another domain's rules
    produces a plausible-looking but meaningless report.
    """
    kyc = run_report(transcripts=REPO_ROOT / "fixtures" / "kyc_calls.jsonl",
                     domain="kyc", settings=load_settings(offline=True))
    keys = {r.cluster.key for r in kyc.coverage.rows}
    collections_only = {"borrower_deceased", "partial_payment_request", "promise_to_pay",
                        "payment_reminder_cooperative", "already_paid", "disputes_amount"}
    assert not (keys & collections_only), (
        f"collections situations leaked into the KYC report: {keys & collections_only}")
    # ...and the KYC-specific ones are present.
    assert {"minor_applicant", "aadhaar_masking_demand"} & keys


def test_critical_flags_are_derived_not_hand_maintained():
    """Critical flags are derived from the assertion table, not maintained separately."""
    assert frozenset(f"compliance:{f}" for f in FLAG_EXPECTATIONS) == CRITICAL_FLAGS


def test_every_compliance_assertion_is_gradeable():
    for flag, exps in FLAG_EXPECTATIONS.items():
        for e in exps:
            assert e.type in _GRADERS, f"{flag} maps to ungradeable {e.type!r}"


def test_unknown_domain_fails_with_a_useful_message(tmp_path):
    load_domain.cache_clear()
    with pytest.raises(ConfigError) as exc:
        load_domain("nope", directory=tmp_path)
    assert "available" in str(exc.value)


def test_a_bad_fail_probability_is_rejected(tmp_path):
    (tmp_path / "broken.yaml").write_text(
        "name: broken\nrule_order: [x]\nsituations:\n  x:\n    weight: 1\n"
        "    fail: 1.5\n    caller: [a]\n    agent: [b]\n")
    load_domain.cache_clear()
    with pytest.raises(ConfigError) as exc:
        load_domain("broken", directory=tmp_path)
    assert "probability" in str(exc.value)
