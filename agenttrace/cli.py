"""Command line interface.

`report`, `diff`, `close-gap`, `agreement` and `gate`. The `gate` subcommand is what runs
in CI and exits non-zero on a coverage or compliance regression.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .agreement import compare_labels
from .config import REPO_ROOT, load_settings
from .coverage import Status
from .errors import AgentTraceError
from .generate import generate_scenario
from .report import RunArtifacts, run_report
from .suite import write_scenario
from .versions import MIN_SAMPLES_PER_SIDE, Verdict, compare_versions

console = Console()


def _fmt_inr(x: float) -> str:
    return f"Rs {x:,.2f}"


def _bar(frac: float, width: int = 12) -> str:
    filled = int(round(frac * width))
    return "#" * filled + "." * (width - filled)


def render(art: RunArtifacts, *, top: int = 12) -> None:
    cov = art.coverage
    s = cov.to_dict()["summary"]

    console.print()
    console.rule("[bold]AgentTrace - coverage report[/bold]")
    console.print(
        f"domain [bold]{art.domain}[/bold]  |  "
        f"corpus [bold]{s['total_conversations']}[/bold] conversations  |  "
        f"suite [bold]{s['suite_size']}[/bold] declared scenarios  |  "
        f"labeler [bold]{s['labeler']}[/bold]"
        + ("  [yellow](DEGRADED)[/yellow]" if s["degraded"] else ""))

    # Two numbers: "declared" is traffic the suite names at all, "fully covered" also
    # requires the conditions production shows to be tested.
    dec, full = s["declared_pct"], s["coverage_pct"]
    dcol = "green" if dec >= 85 else "yellow" if dec >= 65 else "red"
    fcol = "green" if full >= 85 else "yellow" if full >= 65 else "red"
    console.print(
        f"\n[bold]declared      [{dcol}]{dec:5.1f}%[/{dcol}][/bold]  {_bar(dec/100, 24)}"
        f"  in a situation the suite names\n"
        f"[bold]fully covered [{fcol}]{full:5.1f}%[/{fcol}][/bold]  {_bar(full/100, 24)}"
        f"  ...whose conditions are also tested\n"
        f"\n  {s['covered']} covered, [yellow]{s['partial']} partial[/yellow] "
        f"({s['partial_volume']} calls: declared but under-tested), "
        f"[bold red]{s['uncovered']} UNCOVERED[/bold red] "
        f"({s['uncovered_volume']} calls nothing tests)\n"
        f"  unaddressed spend: [bold]{_fmt_inr(s['unaddressed_spend_inr'])}[/bold] "
        f"on failing calls in gap situations")

    t = Table(title="\nRanked coverage gaps", show_lines=False, title_justify="left")
    t.add_column("#", width=3, justify="right")
    t.add_column("situation", style="bold", max_width=30)
    t.add_column("status", width=9)
    t.add_column("vol", justify="right", width=5)
    t.add_column("fail", justify="right", width=6)
    t.add_column("failed Rs", justify="right", width=10)
    t.add_column("flags", max_width=26)
    t.add_column("prio", justify="right", width=7)

    for i, row in enumerate(cov.ranked_gaps(top), 1):
        cl = row.cluster
        st = ("[red]UNCOVERED[/red]" if row.status is Status.UNCOVERED
              else "[yellow]partial[/yellow]")
        flags = ", ".join(row.compliance_flags[:2]) or "-"
        if row.is_critical:
            flags = f"[bold red]{flags}[/bold red]"
        t.add_row(str(i), cl.label, st, str(cl.volume),
                  f"{100*cl.fail_rate:.0f}%", f"{cl.failed_cost_inr:,.0f}",
                  flags, f"{row.priority:.4f}")
    console.print(t)

    covered = Table(title="\nCovered clusters", title_justify="left")
    covered.add_column("situation", style="bold", max_width=30)
    covered.add_column("scenarios")
    covered.add_column("vol", justify="right")
    covered.add_column("fail", justify="right")
    for row in sorted(cov.covered, key=lambda r: -r.cluster.volume)[:10]:
        covered.add_row(row.cluster.label, ",".join(s.id for s in row.matched),
                        str(row.cluster.volume), f"{100*row.cluster.fail_rate:.0f}%")
    console.print(covered)

    if art.grades:
        failed = [g for g in art.grades.values() if not g.passed]
        crit = [g for g in failed if g.has_critical_failure]
        console.print(f"\n[bold]suite assertions[/bold]: graded {len(art.grades)} calls, "
                      f"{len(art.grades)-len(failed)} passed, [red]{len(failed)} failed[/red]"
                      f"  ({len(crit)} with a CRITICAL failure)")
        if crit:
            console.print("  first critical:", crit[0].summary()[:150])

    for note in s["notes"]:
        console.print(f"\n[dim]note: {note}[/dim]")
    console.print()


def _progress_printer(total_hint: str = ""):
    """Single-line progress for long labeling runs."""
    state = {"cost": 0.0, "cached": 0, "failed": 0}

    def cb(i, total, res):
        state["cost"] += res.cost_inr
        state["cached"] += int(res.from_cache)
        state["failed"] += int(not res.ok)
        console.print(
            f"  [{i:>4}/{total}] Rs {state['cost']:7.3f} spent  "
            f"{state['cached']:>4} cached  {state['failed']:>3} failed  "
            f"{res.conversation_id}", end="\r", highlight=False)
        if i == total:
            console.print()
    return cb


def cmd_report(args) -> int:
    settings = load_settings(offline=args.offline or None)
    art = run_report(transcripts=args.transcripts, suite_dir=args.suite,
                     settings=settings, use_llm=args.llm, cost_mode=args.cost_mode,
                     min_cluster_size=args.min_cluster_size, domain=args.domain,
                     converge=args.converge,
                     progress=_progress_printer() if args.progress else None)
    if args.json:
        payload = art.coverage.to_dict()
        payload["ingest"] = art.ingest.summary()
        payload["labels"] = art.labels.summary()
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[dim]wrote {args.json}[/dim]")
    render(art, top=args.top)
    return 0


def cmd_inspect(args) -> int:
    """Everything known about one conversation: transcript, label, grade, cluster, cost."""
    from .costs import component_cost

    settings = load_settings(offline=args.offline or None)
    art = run_report(transcripts=args.transcripts, suite_dir=args.suite,
                     settings=settings, use_llm=args.llm, domain=args.domain)

    conv = next((c for c in art.conversations if c.id == args.call_id), None)
    if conv is None:
        console.print(f"[red]no conversation {args.call_id!r} in {args.transcripts}[/red]")
        return 2

    label = art.labels.labels.get(conv.id)
    grade = art.grades.get(conv.id)
    row = next((r for r in art.coverage.rows if conv.id in r.cluster.conversation_ids), None)

    console.print()
    console.rule(f"[bold]{conv.id}[/bold]")
    console.print(f"\n  {conv.agent_version} · {conv.language} · {round(conv.duration_s)}s · "
                  f"{conv.turn_count} turns · platform said "
                  f"[bold]{conv.disposition or '-'}[/bold]")
    console.print("  tools: " + (", ".join(
        f"{tc.name}{'' if tc.ok else ' (failed)'}" for tc in conv.tool_calls) or "none"))

    console.print("\n[bold]transcript[/bold] [dim](PII redacted)[/dim]")
    for line in conv.transcript().split("\n"):
        who, _, text = line.partition(":")
        colour = "cyan" if who.strip() == "CALLER" else "white"
        console.print(f"  [{colour}]{who:<6}[/{colour}] {text.strip()}")

    console.print(f"\n[bold]what the labeler concluded[/bold] "
                  f"[dim]({art.labeler_by_conversation().get(conv.id, 'n/a')})[/dim]")
    if label is None:
        console.print("  [dim]no label[/dim]")
    else:
        console.print(f"  situation      {label.situation}"
                      + ("  [yellow](invented -- not declared in the suite)[/yellow]"
                         if label.is_new_situation else ""))
        console.print("  agent failed   "
                      + ("[red]yes[/red]" if label.agent_failed else "[green]no[/green]"))
        if label.failure_mode:
            console.print(f"  why            {label.failure_mode}")
        if label.conditions:
            console.print(f"  conditions     {', '.join(label.conditions)}")
        if label.compliance_flags:
            console.print(f"  compliance     [red]{', '.join(label.compliance_flags)}[/red]")
        console.print(f"  confidence     {label.confidence}")

    console.print("\n[bold]what the declared suite asserts[/bold]")
    if grade is None:
        console.print("  [yellow]not graded[/yellow] -- no declared scenario covers this "
                      "situation, so there\n  are no expectations to run. The finding is "
                      "the missing scenario.")
    else:
        console.print(f"  scenario {grade.scenario_id}: "
                      + ("[green]PASS[/green]" if grade.passed else "[red]FAIL[/red]"))
        for c in grade.checks:
            mark = "[green]ok  [/green]" if c.passed else "[red]FAIL[/red]"
            console.print(f"    {mark} {c.type:<28} {c.detail}")

    if row is not None:
        console.print(f"\n[bold]cluster[/bold]  {row.cluster.key}  [{row.status}]  "
                      f"{row.cluster.volume} calls, {100*row.cluster.fail_rate:.0f}% failing")
    console.print(f"\n[dim]cost of this call: "
                  f"{_fmt_inr(component_cost(conv.estimated_usage()).total)} "
                  f"(estimated from the transcript)[/dim]\n")
    return 0


def cmd_agreement(args) -> int:
    """Measure how far the report depends on the model.

    Runs both labelers over the same corpus. The LLM side is cached after the first run.
    """
    from .domains import load_domain
    from .ingest.jsonl import JsonlSource
    from .label import HeuristicLabeler, LlmLabeler, label_corpus
    from .llm.client import SarvamChatClient
    from .suite import load_suite

    settings = load_settings()
    pack = load_domain(args.domain)
    suite = load_suite(args.suite)
    convs, _ = JsonlSource(args.transcripts).load()
    known = sorted(suite.situations)

    heur = label_corpus(convs, labeler=HeuristicLabeler(pack), known_situations=known,
                        settings=settings).labels

    client = SarvamChatClient(settings)
    try:
        llm = label_corpus(convs, labeler=LlmLabeler(client), known_situations=known,
                           settings=settings,
                           progress=_progress_printer() if args.progress else None).labels
    finally:
        client.close()

    rep = compare_labels(heur, llm, a_name="heuristic", b_name="sarvam-105b")
    d = rep.to_dict()

    console.print()
    console.rule("[bold]Inter-labeler agreement[/bold]")
    console.print(
        f"\n[dim]Two labelers with independent failure modes over {rep.n} conversations. "
        f"Disagreement\nlocalises which judgements are model-dependent.[/dim]\n")

    t = Table(show_header=False, box=None)
    t.add_column("k", style="dim", width=30)
    t.add_column("v")
    t.add_row("situation agreement", f"[bold]{100*rep.situation_agreement:.1f}%[/bold]")
    kcol = ("green" if rep.failure_kappa > 0.6 else
            "yellow" if rep.failure_kappa > 0.4 else "red")
    t.add_row("failure agreement (raw)", f"{100*rep.failure_agreement:.1f}%")
    t.add_row("failure agreement (kappa)",
              f"[bold {kcol}]{rep.failure_kappa:.3f}[/bold {kcol}] "
              f"[dim]-- {rep.kappa_reading}[/dim]")
    t.add_row("failure rate, heuristic", f"{100*rep.a_failure_rate:.1f}%")
    t.add_row("failure rate, sarvam-105b", f"{100*rep.b_failure_rate:.1f}%")
    t.add_row("new-situation rate, heuristic", f"{100*rep.a_new_rate:.1f}%")
    t.add_row("new-situation rate, sarvam-105b", f"{100*rep.b_new_rate:.1f}%")
    console.print(t)

    console.print(f"\n[bold]{d['disagreements']['total']}[/bold] conversations disagree: "
                  f"{d['disagreements'].get('situation_only', 0)} on situation only, "
                  f"{d['disagreements'].get('failure_only', 0)} on failure only, "
                  f"{d['disagreements'].get('both', 0)} on both")
    console.print(
        "\n[dim]Kappa corrects for chance agreement: on a corpus that is ~70% "
        "non-failures,\ntwo labelers always answering 'not failed' would score ~70% "
        "raw.[/dim]")

    if rep.b_only_situations:
        console.print("\n[bold]Situations found only by the LLM labeler[/bold]")
        for sit in sorted(rep.b_only_situations)[:12]:
            console.print(f"  - {sit}")
    if args.json:
        Path(args.json).write_text(json.dumps(d, indent=2), encoding="utf-8")
        console.print(f"\n[dim]wrote {args.json}[/dim]")
    console.print()
    return 0


def cmd_close_gap(args) -> int:
    """Generate scenarios for the top uncovered clusters."""
    settings = load_settings(offline=args.offline or None)
    art = run_report(transcripts=args.transcripts, suite_dir=args.suite,
                     settings=settings, use_llm=args.llm, domain=args.domain)
    gaps = [r for r in art.coverage.ranked_gaps() if r.status is Status.UNCOVERED]
    if args.cluster:
        gaps = [r for r in gaps if r.cluster.key == args.cluster]
        if not gaps:
            console.print(f"[red]no uncovered cluster named {args.cluster!r}[/red]")
            return 2
    gaps = gaps[: args.n]

    client = None
    if args.llm and not settings.offline:
        from .llm.client import SarvamChatClient
        client = SarvamChatClient(settings)

    by_id = {c.id: c for c in art.conversations}
    try:
        for row in gaps:
            sc = generate_scenario(row, by_id, client=client)
            console.print(f"\n[bold]{row.cluster.key}[/bold]  "
                          f"({row.cluster.volume} calls, "
                          f"{100*row.cluster.fail_rate:.0f}% failing"
                          + (", [red]COMPLIANCE[/red]" if row.is_critical else "") + ")")
            console.print(f"  -> {sc.id} {sc.name}  [{sc.priority}]")
            for e in sc.expectations:
                bits = [e.type]
                if e.tool:
                    bits.append(f"tool={e.tool}")
                if e.values:
                    bits.append(f"values={e.values}")
                if e.value is not None:
                    bits.append(f"value={e.value}")
                if e.pattern:
                    bits.append(f"pattern={e.pattern!r}")
                console.print(f"     - {' '.join(bits)}")
                if e.reason:
                    console.print(f"       [dim]why: {e.reason}[/dim]")
            if args.write:
                path = write_scenario(sc, args.suite)
                console.print(f"  [green]wrote {path.name}[/green]")
    finally:
        if client:
            client.close()

    if not args.write:
        console.print("\n[dim]Dry run. Pass --write to add these to the suite as files "
                      "for review.[/dim]")
    return 0


def cmd_diff(args) -> int:
    """Compare two agent versions cluster by cluster."""
    settings = load_settings(offline=args.offline or None)
    art = run_report(transcripts=args.transcripts, suite_dir=args.suite,
                     settings=settings, use_llm=args.llm, domain=args.domain,
                     min_cluster_size=args.min_cluster_size)
    cmp = compare_versions(art.coverage, args.baseline, args.candidate)
    o = cmp.overall_rates()

    console.print()
    console.rule(f"[bold]Version diff: {args.baseline} -> {args.candidate}[/bold]")
    console.print(
        f"\naggregate failure rate  "
        f"{100*o['baseline']['rate']:.1f}%  ->  {100*o['candidate']['rate']:.1f}%   "
        f"(delta {100*o['aggregate_delta']:+.1f}pp over "
        f"{o['baseline']['n']} vs {o['candidate']['n']} calls)")
    console.print(
        "[dim]The aggregate does not identify which situation moved; see the per-cluster\n"
        "table below.[/dim]")

    t = Table(title="\nPer-cluster comparison", title_justify="left")
    t.add_column("situation", style="bold", max_width=30)
    t.add_column("verdict", width=18)
    t.add_column(args.baseline, justify="right", width=14)
    t.add_column(args.candidate, justify="right", width=14)
    t.add_column("delta", justify="right", width=8)
    t.add_column("p", justify="right", width=9)

    style = {Verdict.REGRESSION: "bold red", Verdict.IMPROVEMENT: "green",
             Verdict.NEW_CLUSTER: "yellow", Verdict.DISAPPEARED: "yellow",
             Verdict.UNCHANGED: "dim", Verdict.INSUFFICIENT_DATA: "dim"}
    for d in cmp.diffs:
        st = style[d.verdict]
        name = str(d.verdict).upper() if d.verdict is Verdict.REGRESSION else str(d.verdict)
        t.add_row(
            d.label, f"[{st}]{name}[/{st}]",
            f"{d.baseline_failures}/{d.baseline_n} ({100*d.baseline_rate:.0f}%)"
            if d.baseline_n else "-",
            f"{d.candidate_failures}/{d.candidate_n} ({100*d.candidate_rate:.0f}%)"
            if d.candidate_n else "-",
            f"{100*d.delta:+.0f}pp" if d.baseline_n and d.candidate_n else "-",
            f"{d.p_value:.4f}" if d.verdict in
            {Verdict.REGRESSION, Verdict.IMPROVEMENT, Verdict.UNCHANGED} else "-")
    console.print(t)

    if cmp.regressions:
        console.print(f"\n[bold red]{len(cmp.regressions)} STATISTICALLY SIGNIFICANT "
                      f"REGRESSION(S)[/bold red]")
        for d in cmp.regressions:
            console.print(f"  - {d.explain()}")
        console.print("\n[dim]Remaining clusters are unchanged or below the sample floor "
                      f"of {MIN_SAMPLES_PER_SIDE} per side.[/dim]")
    else:
        console.print("\n[green]no statistically significant regressions[/green]")
    console.print()
    return 1 if (cmp.should_block and args.fail_on_regression) else 0


def cmd_gate(args) -> int:
    """Exit non-zero if coverage or compliance regressed."""
    settings = load_settings(offline=args.offline or None)
    art = run_report(transcripts=args.transcripts, suite_dir=args.suite,
                     settings=settings, use_llm=args.llm, domain=args.domain)
    cov = art.coverage
    problems: list[str] = []

    # Two thresholds: undeclared traffic needs a new scenario, a partial cluster needs an
    # existing one extended.
    if cov.declared_pct < args.min_declared:
        problems.append(f"declared coverage {cov.declared_pct:.1f}% < required "
                        f"{args.min_declared:.1f}% ({cov.uncovered_volume} calls in "
                        f"situations no scenario names)")
    if cov.coverage_pct < args.min_coverage:
        problems.append(f"fully-covered {cov.coverage_pct:.1f}% < required "
                        f"{args.min_coverage:.1f}%")
    crit = [r for r in cov.ranked_gaps() if r.is_critical]
    if crit:
        problems.append(f"{len(crit)} uncovered cluster(s) carry compliance exposure: "
                        + ", ".join(r.cluster.key for r in crit[:4]))
    failed_critical = [g for g in art.grades.values() if g.has_critical_failure]
    if failed_critical:
        problems.append(f"{len(failed_critical)} graded calls failed a CRITICAL assertion")

    if problems:
        console.print("[bold red]GATE FAILED[/bold red]")
        for p in problems:
            console.print(f"  - {p}")
        return 1
    console.print("[bold green]GATE PASSED[/bold green]")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("agenttrace", description="Coverage and regression analysis "
                                                     "for voice agents.")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("-v", "--verbose", action="store_true",
                        help="also accepted before the subcommand")
        sp.add_argument("--domain", default="collections",
                        help="domain pack; selects the taxonomy, the suite and the corpus")
        sp.add_argument("--converge", action="store_true",
                        help="show the labeler canonical slugs from previous runs to reduce "
                             "synonym fragmentation (costs a fresh labeling pass)")
        sp.add_argument("--progress", action="store_true",
                        help="print per-conversation labeling progress")
        # None so --domain resolves them from the pack; an explicit value still wins.
        sp.add_argument("--transcripts", type=Path, default=None)
        sp.add_argument("--suite", type=Path, default=None)
        sp.add_argument("--llm", action="store_true",
                        help="use Sarvam-105B for labeling (default: heuristic, offline)")
        sp.add_argument("--offline", action="store_true", help="never touch the network")

    r = sub.add_parser("report", help="print the coverage report")
    common(r)
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--json", type=Path, help="also write the report as JSON")
    r.add_argument("--cost-mode", choices=["component", "platform"], default="component")
    r.add_argument("--min-cluster-size", type=int, default=3)
    r.set_defaults(fn=cmd_report)

    ins = sub.add_parser("inspect", help="everything known about one conversation")
    common(ins)
    ins.add_argument("call_id")
    ins.set_defaults(fn=cmd_inspect)

    ag = sub.add_parser("agreement",
                        help="measure how far the report depends on the model")
    common(ag)
    ag.add_argument("--json", type=Path)
    ag.set_defaults(fn=cmd_agreement)

    cg = sub.add_parser("close-gap", help="generate scenarios for the top uncovered clusters")
    common(cg)
    cg.add_argument("-n", type=int, default=1)
    cg.add_argument("--cluster", help="close this specific cluster key")
    cg.add_argument("--write", action="store_true", help="write the YAML into the suite")
    cg.set_defaults(fn=cmd_close_gap)

    d = sub.add_parser("diff", help="compare two agent versions cluster by cluster")
    common(d)
    d.add_argument("--baseline", default="v2")
    d.add_argument("--candidate", default="v3")
    d.add_argument("--min-cluster-size", type=int, default=3)
    d.add_argument("--fail-on-regression", action="store_true",
                   help="exit 1 if any significant regression is found (for CI)")
    d.set_defaults(fn=cmd_diff)

    g = sub.add_parser("gate", help="exit non-zero if coverage or compliance regressed")
    common(g)
    g.add_argument("--min-coverage", type=float, default=60.0,
                   help="minimum share of traffic FULLY covered")
    g.add_argument("--min-declared", type=float, default=85.0,
                   help="minimum share of traffic in a situation the suite names")
    g.set_defaults(fn=cmd_gate)

    return p


def _resolve_domain_paths(args) -> None:
    """Fill --transcripts/--suite from the domain pack when not given explicitly."""
    from .domains import load_domain
    pack = load_domain(args.domain)
    if args.transcripts is None:
        args.transcripts = REPO_ROOT / "fixtures" / f"{pack.name}_calls.jsonl"
    if args.suite is None:
        args.suite = pack.suite_dir()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if hasattr(args, "domain"):
        _resolve_domain_paths(args)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    try:
        return args.fn(args)
    except AgentTraceError as exc:
        console.print(f"[bold red]{type(exc).__name__}[/bold red]: {exc}")
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
