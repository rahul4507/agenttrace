"""Tests for the coverage diff, graders, clustering and version attribution.

These cover the meaning of the report rather than the plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agenttrace.cluster import build_clusters, normalise
from agenttrace.coverage import Status, build_coverage
from agenttrace.errors import SuiteError
from agenttrace.graders import grade
from agenttrace.label import SituationLabel
from agenttrace.models import Conversation, Outcome, Role, ToolCall, Turn
from agenttrace.redact import contains_pii, redact
from agenttrace.suite import Expectation, Scenario, Suite, load_suite
from agenttrace.versions import (
    Verdict,
    compare_versions,
    fisher_exact_greater,
    wilson_interval,
)


def conv(cid="c1", *, version="v2", turns=None, outcome=None, situation=None) -> Conversation:
    return Conversation(
        id=cid, agent_id="a", agent_version=version, duration_s=60.0,
        turns=turns or [Turn(role=Role.AGENT, text="Namaste")],
        outcome=outcome,
        metadata={"_truth_situation": situation} if situation else {})


def label(situation, *, failed=False, conditions=(), flags=(), new=False) -> SituationLabel:
    return SituationLabel(
        situation=situation, situation_label=situation.replace("_", " "),
        is_new_situation=new, conditions=list(conditions), agent_failed=failed,
        failure_mode="broke" if failed else None,
        compliance_flags=list(flags), confidence=0.9)


def scenario(sid, situation, *, expectations=(), conditions=()) -> Scenario:
    return Scenario(id=sid, name=sid, situation=situation,
                    expectations=list(expectations), conditions=list(conditions))


# Coverage as a diff

def test_situation_in_suite_is_covered():
    convs = [conv(f"c{i}") for i in range(10)]
    labels = {c.id: label("already_paid") for c in convs}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    assert [r.status for r in rep.rows] == [Status.COVERED]
    assert rep.coverage_pct == 100.0


def test_situation_absent_from_suite_is_uncovered():
    convs = [conv(f"c{i}") for i in range(10)]
    labels = {c.id: label("borrower_deceased", failed=True) for c in convs}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    assert rep.uncovered[0].cluster.key == "borrower_deceased"
    assert rep.coverage_pct == 0.0


def test_coverage_is_traffic_weighted_not_cluster_weighted():
    """A suite covering one of two clusters can still cover 90% of calls."""
    convs = [conv(f"big{i}") for i in range(90)] + [conv(f"small{i}") for i in range(10)]
    labels = {**{f"big{i}": label("already_paid") for i in range(90)},
              **{f"small{i}": label("borrower_deceased") for i in range(10)}}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    assert rep.coverage_pct == pytest.approx(90.0)
    assert len(rep.uncovered) == 1          # half the CLUSTERS are uncovered...
    assert rep.uncovered_volume == 10       # ...but only a tenth of the TRAFFIC


def test_declared_situation_with_untested_conditions_is_partial():
    """Naming the right situation is not the same as testing its conditions."""
    convs = [conv(f"c{i}") for i in range(20)]
    labels = {c.id: label("already_paid", conditions=["angry_caller"]) for c in convs}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail,
                         Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    assert rep.rows[0].status is Status.PARTIAL
    assert "angry_caller" in rep.rows[0].missing_conditions


def test_covered_cluster_never_gets_a_gap_priority():
    convs = [conv(f"c{i}") for i in range(10)]
    labels = {c.id: label("already_paid", failed=(i < 5)) for i, c in enumerate(convs)}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    covered = [r for r in rep.rows if r.status is Status.COVERED]
    assert all(r.priority == 0.0 for r in covered)


# Compliance thresholds

def test_one_bad_call_does_not_make_a_cluster_a_compliance_cluster():
    """Three compliance hits in 84 calls is not a compliance cluster.

    Without a rate threshold, a handful of bad calls flags an otherwise healthy cluster and
    every row on the report reads as critical.
    """
    convs = [conv(f"c{i}") for i in range(84)]
    labels = {c.id: label("already_paid",
                          flags=["third_party_disclosure"] if i < 3 else [])
              for i, c in enumerate(convs)}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    row = rep.rows[0]
    assert not row.is_critical
    assert row.compliance_flags == []
    # ...but it is disclosed as a below-threshold observation, not silently dropped.
    assert any(f["flag"] == "third_party_disclosure" for f in row.suppressed_flags)


def test_a_genuine_compliance_cluster_is_flagged():
    convs = [conv(f"c{i}") for i in range(20)]
    labels = {c.id: label("third_party_answered", failed=True,
                          flags=["third_party_disclosure"]) for c in convs}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[]), total_conversations=20)
    assert rep.rows[0].is_critical
    assert rep.rows[0].compliance_detail[0]["rate"] == 1.0


def test_small_cluster_still_flagged_above_absolute_floor():
    """A small but severe cluster is not dismissed for lack of volume."""
    convs = [conv(f"c{i}") for i in range(4)]
    labels = {c.id: label("borrower_deceased", failed=True,
                          flags=["deceased_borrower_pursued"]) for c in convs}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[]), total_conversations=4)
    assert rep.rows[0].is_critical


# --- clustering ------------------------------------------------------------------

def test_open_set_synonyms_merge_into_one_cluster():
    """Open-set labeling emits synonyms; they must not be reported as separate gaps."""
    convs = [conv(f"c{i}") for i in range(9)]
    variants = ["disputes_amount", "amount_dispute", "disputed_amount"]
    labels = {f"c{i}": label(variants[i % 3]) for i in range(9)}
    clusters, _ = build_clusters(convs, labels, min_cluster_size=3)
    assert len(clusters) == 1
    assert clusters[0].volume == 9
    assert len(clusters[0].member_slugs) == 1  # all alias to the same canonical slug


def test_tail_clusters_are_returned_not_discarded():
    convs = [conv(f"c{i}") for i in range(12)]
    # Unrelated slugs: at the 0.5 threshold anything sharing two tokens would merge.
    labels = {**{f"c{i}": label("already_paid") for i in range(10)},
              "c10": label("courier_delivery_query"), "c11": label("branch_timings")}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    assert len(clusters) == 1
    assert len(tail) == 2, "the long tail must be visible, not dropped"


def test_normalise_is_idempotent():
    for s in ["Disputes_Amount", "disputes-amount", "amount_dispute"]:
        assert normalise(normalise(s)) == normalise(s)


# --- graders ---------------------------------------------------------------------

def test_verify_before_disclosure_catches_ordering_not_just_presence():
    """Ordering is the requirement, so a set-membership check is insufficient."""
    c = conv(turns=[
        Turn(role=Role.AGENT, text="Aapka EMI 4,500 rupaye pending hai"),   # disclose
        Turn(role=Role.CALLER, text="kaun bol raha hai"),
        Turn(role=Role.AGENT, text="verify karte hain",
             tool_calls=[ToolCall(name="verify_identity", ok=True)]),        # too late
    ])
    g = grade(c, scenario("S", "x", expectations=[
        Expectation(type="verify_before_disclosure")]))
    assert not g.passed and g.has_critical_failure


def test_verify_before_disclosure_passes_in_correct_order():
    c = conv(turns=[
        Turn(role=Role.AGENT, text="Namaste", tool_calls=[ToolCall(name="verify_identity", ok=True)]),
        Turn(role=Role.AGENT, text="Aapka EMI 4,500 rupaye pending hai"),
    ])
    g = grade(c, scenario("S", "x", expectations=[Expectation(type="verify_before_disclosure")]))
    assert g.passed


def test_a_failed_verification_tool_call_does_not_count_as_verified():
    c = conv(turns=[
        Turn(role=Role.AGENT, text="Namaste",
             tool_calls=[ToolCall(name="verify_identity", ok=False, error="timeout")]),
        Turn(role=Role.AGENT, text="Aapka EMI 4,500 rupaye pending hai"),
    ])
    g = grade(c, scenario("S", "x", expectations=[Expectation(type="verify_before_disclosure")]))
    assert not g.passed


def test_must_not_say_catches_hinglish_legal_threats():
    """The prohibited-phrase list must match Hinglish, not only English."""
    c = conv(turns=[Turn(role=Role.AGENT, text="Recovery team aapke ghar aa sakti hai")])
    g = grade(c, scenario("S", "x", expectations=[Expectation(type="must_not_say")]))
    assert not g.passed and g.has_critical_failure


def test_repeated_agent_turn_detects_the_loop():
    same = "Maaf kijiye, main samajh nahi payi. Dobara boliye."
    c = conv(turns=[Turn(role=Role.AGENT, text=same) for _ in range(3)])
    g = grade(c, scenario("S", "x", expectations=[Expectation(type="no_repeated_agent_turn")]))
    assert not g.passed


def test_caller_hangup_is_not_graded_as_an_agent_failure():
    """Caller-side endings are not agent failures."""
    assert not Outcome.CALLER_ABANDONED.is_failure
    assert not Outcome.NOT_CONNECTED.is_failure
    assert Outcome.AGENT_ERROR.is_failure
    assert Outcome.COMPLIANCE_BREACH.is_failure
    assert Outcome.ESCALATED.is_success, "a correct handoff is a success, not a failure"


# --- version attribution ---------------------------------------------------------

def _report_with_versions(baseline_fail, baseline_n, cand_fail, cand_n):
    convs, labels = [], {}
    for i in range(baseline_n):
        convs.append(conv(f"b{i}", version="v2"))
        labels[f"b{i}"] = label("already_paid", failed=i < baseline_fail)
    for i in range(cand_n):
        convs.append(conv(f"c{i}", version="v3"))
        labels[f"c{i}"] = label("already_paid", failed=i < cand_fail)
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    return build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                          total_conversations=len(convs))


def test_significant_regression_is_detected():
    cmp = compare_versions(_report_with_versions(1, 25, 10, 21), "v2", "v3")
    d = cmp.diffs[0]
    assert d.verdict is Verdict.REGRESSION
    assert d.p_value < 0.01
    assert cmp.should_block


def test_noise_is_not_reported_as_a_regression():
    """2/18 -> 5/18 is a 167% relative move and consistent with noise."""
    cmp = compare_versions(_report_with_versions(2, 18, 5, 18), "v2", "v3")
    assert cmp.diffs[0].verdict is Verdict.UNCHANGED
    assert not cmp.should_block


def test_small_samples_return_insufficient_data_not_a_verdict():
    cmp = compare_versions(_report_with_versions(0, 5, 3, 5), "v2", "v3")
    assert cmp.diffs[0].verdict is Verdict.INSUFFICIENT_DATA
    assert not cmp.should_block


def test_improvement_is_distinguished_from_regression():
    cmp = compare_versions(_report_with_versions(20, 30, 2, 30), "v2", "v3")
    assert cmp.diffs[0].verdict is Verdict.IMPROVEMENT
    assert not cmp.should_block


def test_aggregate_can_improve_while_a_cluster_regresses():
    """An aggregate can improve while an individual cluster regresses."""
    convs, labels = [], {}
    # A big cluster gets much better...
    for i in range(60):
        convs.append(conv(f"bigb{i}", version="v2"))
        labels[f"bigb{i}"] = label("promise_to_pay", failed=i < 30)
    for i in range(60):
        convs.append(conv(f"bigc{i}", version="v3"))
        labels[f"bigc{i}"] = label("promise_to_pay", failed=i < 6)
    # ...while a smaller one regresses badly.
    for i in range(20):
        convs.append(conv(f"smb{i}", version="v2"))
        labels[f"smb{i}"] = label("already_paid", failed=i < 1)
    for i in range(20):
        convs.append(conv(f"smc{i}", version="v3"))
        labels[f"smc{i}"] = label("already_paid", failed=i < 12)

    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[
        scenario("SC-1", "already_paid"), scenario("SC-2", "promise_to_pay")]),
        total_conversations=len(convs))
    cmp = compare_versions(rep, "v2", "v3")

    assert cmp.overall_rates()["aggregate_delta"] < 0, "aggregate improved"
    regressed = {d.key for d in cmp.regressions}
    assert "already_paid" in regressed, "yet a cluster regressed and must be caught"
    assert cmp.should_block


def test_fisher_is_symmetric_under_no_change():
    assert fisher_exact_greater(5, 35, 5, 35) > 0.5


def test_wilson_interval_never_collapses_at_zero():
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and hi > 0.05, "Wald would give a zero-width interval here"


# --- suite integrity -------------------------------------------------------------

def test_unknown_expectation_type_is_a_hard_error(tmp_path):
    """An unknown expectation type must raise rather than assert nothing."""
    (tmp_path / "bad.yaml").write_text(
        "id: X\nname: x\nsituation: y\nexpectations:\n  - type: tool_calledd\n    tool: t\n")
    with pytest.raises(SuiteError) as exc:
        load_suite(tmp_path)
    assert "expectations" in str(exc.value)


def test_duplicate_scenario_ids_are_rejected(tmp_path):
    for n in ("a.yaml", "b.yaml"):
        (tmp_path / n).write_text("id: SAME\nname: x\nsituation: y\n")
    with pytest.raises(SuiteError) as exc:
        load_suite(tmp_path)
    assert "duplicate" in str(exc.value)


def test_stray_key_in_a_scenario_is_rejected(tmp_path):
    (tmp_path / "a.yaml").write_text("id: X\nname: x\nsituation: y\nprioriy: P0\n")
    with pytest.raises(SuiteError):
        load_suite(tmp_path)


def test_the_shipped_suite_loads_and_every_expectation_is_gradeable():
    from agenttrace.config import REPO_ROOT
    from agenttrace.graders import _GRADERS
    suite = load_suite(REPO_ROOT / "suite" / "collections")
    assert len(suite) >= 8
    for sc in suite.scenarios:
        for e in sc.expectations:
            assert e.type in _GRADERS, f"{sc.id} declares ungradeable {e.type!r}"


# --- redaction -------------------------------------------------------------------

@pytest.mark.parametrize("text,expect_tag", [
    ("card 4111 1111 1111 1111 hai", "CARD"),
    ("Aadhaar 4123 5678 9012", "AADHAAR"),
    ("PAN ABCDE1234F", "PAN"),
    ("number 98765 43210", "PHONE"),
    ("IFSC HDFC0001234", "IFSC"),
    ("UPI rahul@okhdfcbank", "UPI"),
])
def test_redaction_tags_indian_pii(text, expect_tag):
    assert f"[{expect_tag}]" in redact(text)
    assert expect_tag in contains_pii(text)


def test_card_number_is_not_eaten_by_the_aadhaar_rule():
    """A 16-digit card contains a 12-digit Aadhaar-shaped prefix."""
    out = redact("card 4111 1111 1111 1111 charge karo")
    assert out == "card [CARD] charge karo"
    assert "1111" not in out, "no digits of the card may survive"


def test_bearer_token_is_fully_redacted():
    """`key: Bearer <token>` must redact the token, not just the keyword."""
    out = redact("Authorization: Bearer eyJhbGci.OiJIUzI1.NiIsInR5")
    assert "eyJhbGci" not in out


def test_a_phone_is_not_double_tagged_as_an_account():
    assert contains_pii("9876543210 par call karein") == ["PHONE"]


def test_amounts_are_not_redacted():
    """Amounts must survive redaction; the transcripts are analysed on them."""
    assert redact("EMI Rs 4500 due hai") == "EMI Rs 4500 due hai"


def test_two_declared_situations_never_merge_into_each_other():
    """Similarly named scenarios can still be different tests.

    `refund_status_dispute` and `refund_status_confirmation` share two tokens (Jaccard
    0.67), so they merge at the 0.5 threshold. If both are declared, merging costs each its
    own coverage verdict, so the protected-slug guard keeps them distinct.
    """
    convs = [conv(f"a{i}") for i in range(10)] + [conv(f"b{i}") for i in range(10)]
    labels = {**{f"a{i}": label("refund_status_dispute") for i in range(10)},
              **{f"b{i}": label("refund_status_confirmation") for i in range(10)}}

    # Without the guard they merge.
    merged, _ = build_clusters(convs, labels, min_cluster_size=3)
    assert len(merged) == 1

    protected = frozenset({"refund_status_dispute", "refund_status_confirmation"})
    kept, _ = build_clusters(convs, labels, min_cluster_size=3, protected=protected)
    assert len(kept) == 2, "two declared situations must keep separate coverage verdicts"


def test_declared_and_fully_covered_are_reported_as_separate_numbers():
    """`declared_pct` and `coverage_pct` measure different things.

    Traffic can be almost entirely in declared situations while little of it is fully
    covered, so a single figure would be ambiguous.
    """
    convs = [conv(f"c{i}") for i in range(20)]
    # Declared situation, but production shows a condition no scenario tests -> PARTIAL.
    labels = {c.id: label("already_paid", conditions=["angry_caller"]) for c in convs}
    clusters, tail = build_clusters(convs, labels, min_cluster_size=3)
    rep = build_coverage(clusters, tail, Suite(scenarios=[scenario("SC-1", "already_paid")]),
                         total_conversations=len(convs))
    assert rep.rows[0].status is Status.PARTIAL
    assert rep.declared_pct == pytest.approx(100.0), "the situation IS declared"
    assert rep.coverage_pct == pytest.approx(0.0), "but it is not fully covered"
    assert rep.undeclared_pct == pytest.approx(0.0), "and nothing is undeclared"
