from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Literal

from clinidoc.findings import (
    FIX_PLAN,
    SEVERITY_RANK,
    Finding,
    is_trainability,
)
from clinidoc import __version__

Verdict = Literal["pass", "caution", "blocked"]


def compute_verdict(findings: Iterable[Finding], *, fail_on_phi: bool = False) -> Verdict:
    worst = 0
    for item in findings:
        if item.group == "phi" and not fail_on_phi:
            continue
        if item.group == "phi" and fail_on_phi:
            return "blocked"
        if not is_trainability(item) and item.group != "phi":
            continue
        rank = SEVERITY_RANK.get(item.severity, 0)
        if item.severity == "critical":
            return "blocked"
        worst = max(worst, rank)
    if worst >= SEVERITY_RANK["major"] or worst >= SEVERITY_RANK["minor"]:
        return "caution"
    return "pass"


def exit_code(
    verdict: Verdict,
    findings: Iterable[Finding],
    *,
    fail_on: str = "critical",
    fail_on_phi: bool = False,
) -> int:
    items = list(findings)
    if fail_on_phi and any(f.group == "phi" for f in items):
        return 2 if verdict == "blocked" else 1
    if verdict == "blocked":
        return 2
    if fail_on == "major" and verdict == "caution":
        return 1
    return 0


def summarize(findings: Iterable[Finding]) -> dict[str, Any]:
    items = list(findings)
    by_sev = Counter(f.severity for f in items)
    by_group = Counter(f.group for f in items)
    return {
        "count": len(items),
        "by_severity": dict(by_sev),
        "by_group": dict(by_group),
    }


def build_fix_plan(findings: Iterable[Finding]) -> list[dict[str, str]]:
    seen: set[str] = set()
    plan: list[dict[str, str]] = []
    ordered = sorted(findings, key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), f.id))
    for item in ordered:
        if item.id in seen:
            continue
        seen.add(item.id)
        plan.append(
            {
                "id": item.id,
                "severity": item.severity,
                "action": FIX_PLAN.get(item.id, "Inspect and repair the listed documents."),
            }
        )
    return plan


def findings_to_json(
    *,
    path: str,
    detected_format: str,
    n_docs: int,
    findings: list[Finding],
    verdict: Verdict,
) -> dict[str, Any]:
    return {
        "clinicdoc_version": __version__,
        "path": path,
        "format": detected_format,
        "documents": n_docs,
        "verdict": verdict,
        "summary": summarize(findings),
        "findings": [f.to_dict() for f in findings],
        "fix_plan": build_fix_plan(findings),
    }


def findings_to_sarif(findings: list[Finding]) -> dict[str, Any]:
    level = {"critical": "error", "major": "warning", "minor": "note", "governance": "note"}
    from clinidoc.findings import ALL_CHECKS

    rules = [
        {
            "id": spec.id,
            "name": spec.id,
            "shortDescription": {"text": spec.description},
            "defaultConfiguration": {
                "level": level.get(spec.default_severity, "note"),
            },
            "properties": {"tags": [spec.group]},
        }
        for spec in ALL_CHECKS
    ]
    results = []
    for item in findings:
        results.append(
            {
                "ruleId": item.id,
                "level": level.get(item.severity, "note"),
                "message": {"text": item.message},
                "properties": {
                    "group": item.group,
                    "document_id": item.document_id,
                    "split": item.split,
                    "evidence": item.evidence,
                },
            }
        )
    return {
        "version": "2.1.0",
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "clinicdoc",
                        "version": __version__,
                        "informationUri": "https://github.com/SyedAshhadIbrar/Clinicdoc",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }


def markdown_report(
    *,
    path: str,
    detected_format: str,
    n_docs: int,
    splits: dict[str, int],
    findings: list[Finding],
    verdict: Verdict,
) -> str:
    summary = summarize(findings)
    lines = [
        "# Clinicdoc audit",
        "",
        f"**Verdict:** `{verdict}`",
        f"**Path:** `{path}`",
        f"**Format:** `{detected_format}`",
        f"**Documents:** {n_docs}",
        f"**Splits:** {', '.join(f'{k}={v}' for k, v in splits.items()) or '(none)'}",
        f"**Findings:** {summary['count']}",
        "",
        "## Severity",
        "",
        "| Severity | Count |",
        "|---|---|",
    ]
    for sev in ("critical", "major", "minor", "governance"):
        lines.append(f"| {sev} | {summary['by_severity'].get(sev, 0)} |")
    lines.extend(["", "## Findings", ""])
    if not findings:
        lines.append("No findings.")
    else:
        ordered = sorted(findings, key=lambda f: (-SEVERITY_RANK.get(f.severity, 0), f.group, f.id, f.document_id or ""))
        for item in ordered:
            loc = item.document_id or "(corpus)"
            extra = ""
            snippet = item.evidence.get("snippet") if item.evidence else None
            if snippet:
                extra = f" — `{snippet}`"
            lines.append(f"- **{item.severity}** `{item.id}` {loc}: {item.message}{extra}")
    lines.extend(["", "## Fix plan", ""])
    plan = build_fix_plan(findings)
    if not plan:
        lines.append("No actions required.")
    else:
        for i, step in enumerate(plan, start=1):
            lines.append(f"{i}. `{step['id']}` ({step['severity']}): {step['action']}")
    lines.append("")
    return "\n".join(lines)


def write_audit_outputs(
    out_dir: Path,
    *,
    payload: dict[str, Any],
    markdown: str,
    sarif: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.md").write_text(markdown, encoding="utf-8")
    (out_dir / "findings.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    (out_dir / "fix_plan.json").write_text(json.dumps(payload["fix_plan"], indent=2) + "\n", encoding="utf-8")
    (out_dir / "findings.sarif").write_text(json.dumps(sarif, indent=2) + "\n", encoding="utf-8")
