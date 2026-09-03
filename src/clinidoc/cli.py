from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from clinidoc import __version__
from clinidoc.dataset import load_dataset
from clinidoc.detectors import list_checks, scan_all
from clinidoc.findings import SEVERITY_RANK
from clinidoc.report import (
    compute_verdict,
    exit_code,
    findings_to_json,
    findings_to_sarif,
    markdown_report,
    write_audit_outputs,
)
from clinidoc.resplit import ResplitError, resplit

console = Console()
err_console = Console(stderr=True)

VERDICT_STYLE = {
    "pass": "bold green",
    "caution": "bold yellow",
    "blocked": "bold red",
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("path", help="Dataset file or directory")
    parser.add_argument(
        "--input-format",
        dest="input_format",
        choices=["jsonl", "csv", "brat", "conll", "bio"],
        default=None,
        help="Override format auto-detect",
    )
    parser.add_argument(
        "--fail-on",
        choices=["critical", "major"],
        default="critical",
        help="Non-zero exit if findings at this severity or worse (blocked always exits 2)",
    )
    parser.add_argument(
        "--format",
        "--output-format",
        dest="output_format",
        choices=["text", "json", "sarif"],
        default="text",
        help="stdout format (CI: --format sarif|json)",
    )
    parser.add_argument(
        "--fail-on-phi",
        action="store_true",
        help="Treat PHI/governance findings as a training blocker",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clinicdoc",
        description="Audit clinical NLP datasets before training. Offline; note text never leaves the machine.",
    )
    parser.add_argument("--version", action="version", version=f"clinicdoc {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    scan_p = sub.add_parser("scan", help="Run all detectors and print findings")
    _add_common(scan_p)

    audit_p = sub.add_parser("audit", help="Scan, write ranked report, verdict, and fix plan")
    _add_common(audit_p)
    audit_p.add_argument("--out", default="out", help="Directory for report.md, findings.json, fix_plan.json")

    resplit_p = sub.add_parser("resplit", help="Rewrite splits grouped by patient_id; refuse leaking output")
    resplit_p.add_argument("path")
    resplit_p.add_argument("--by", default="patient_id", help="Grouping key (v1: patient_id)")
    resplit_p.add_argument("--out", required=True, help="Output directory for train/val/test jsonl")
    resplit_p.add_argument("--seed", type=int, default=13)
    resplit_p.add_argument("--input-format", dest="input_format", default=None)

    sub.add_parser("detectors", help="List checks and groups")
    return parser


def _print_text(dataset, findings, verdict: str) -> None:
    splits = Counter(d.split for d in dataset.documents)
    split_s = ", ".join(f"{k}={v}" for k, v in sorted(splits.items())) or "none"
    header = Table.grid(padding=(0, 1))
    header.add_column(style="dim")
    header.add_column()
    header.add_row("path", str(dataset.root))
    header.add_row("format", dataset.detected_format)
    header.add_row("documents", str(len(dataset.documents)))
    header.add_row("splits", split_s)
    console.print(header)
    console.print()

    counts = Counter(f.severity for f in findings)
    table = Table(title="Findings", show_lines=False)
    table.add_column("Sev", style="bold")
    table.add_column("Check")
    table.add_column("Where")
    table.add_column("Message")
    ordered = sorted(
        findings,
        key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), f.group, f.id, f.document_id or ""),
    )
    styles = {"critical": "red", "major": "yellow", "minor": "cyan", "governance": "magenta"}
    for item in ordered:
        where = item.document_id or item.split or "—"
        table.add_row(
            f"[{styles.get(item.severity, 'white')}]{item.severity}[/]",
            item.id,
            str(where),
            item.message,
        )
    if findings:
        console.print(table)
    else:
        console.print("[green]No findings.[/green]")
    console.print()
    summary = "  ".join(f"{k}={counts.get(k, 0)}" for k in ("critical", "major", "minor", "governance"))
    console.print(f"[dim]{len(findings)} finding(s): {summary}[/dim]")
    console.print(
        Panel(
            Text(verdict.upper(), style=VERDICT_STYLE.get(verdict, "bold")),
            title="verdict",
            border_style={"pass": "green", "caution": "yellow", "blocked": "red"}.get(verdict, "white"),
        )
    )


def _run_scan(args: argparse.Namespace, *, write_audit: bool) -> int:
    fmt = args.input_format
    if fmt == "bio":
        fmt = "conll"
    dataset = load_dataset(args.path, input_format=fmt)
    findings = scan_all(dataset)
    verdict = compute_verdict(findings, fail_on_phi=args.fail_on_phi)
    splits = dict(Counter(d.split for d in dataset.documents))
    payload = findings_to_json(
        path=str(dataset.root),
        detected_format=dataset.detected_format,
        n_docs=len(dataset.documents),
        findings=findings,
        verdict=verdict,
    )
    sarif = findings_to_sarif(findings)
    md = markdown_report(
        path=str(dataset.root),
        detected_format=dataset.detected_format,
        n_docs=len(dataset.documents),
        splits=splits,
        findings=findings,
        verdict=verdict,
    )
    if write_audit:
        write_audit_outputs(Path(args.out), payload=payload, markdown=md, sarif=sarif)
        err_console.print(f"[dim]Wrote {args.out}/report.md, findings.json, fix_plan.json[/dim]")
    if args.output_format == "json":
        console.print_json(data=payload)
    elif args.output_format == "sarif":
        console.print_json(data=sarif)
    else:
        _print_text(dataset, findings, verdict)
    return exit_code(verdict, findings, fail_on=args.fail_on, fail_on_phi=args.fail_on_phi)


def _run_detectors() -> int:
    table = Table(title="Clinicdoc detectors")
    table.add_column("Group")
    table.add_column("Id")
    table.add_column("Default severity")
    table.add_column("Description")
    for spec in list_checks():
        table.add_row(spec.group, spec.id, spec.default_severity, spec.description)
    console.print(table)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.cmd == "detectors":
            return _run_detectors()
        if args.cmd == "scan":
            return _run_scan(args, write_audit=False)
        if args.cmd == "audit":
            return _run_scan(args, write_audit=True)
        if args.cmd == "resplit":
            _ds, _leaks, written = resplit(
                args.path,
                args.out,
                by=args.by,
                seed=args.seed,
                input_format=args.input_format,
            )
            console.print(f"[green]Wrote non-leaking split to {args.out}[/green]")
            for path in written:
                console.print(f"  {path}")
            return 0
        parser.error(f"unknown command {args.cmd}")
        return 2
    except ResplitError as exc:
        err_console.print(f"[red]resplit refused:[/red] {exc}")
        return 2
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
