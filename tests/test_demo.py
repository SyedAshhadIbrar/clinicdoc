from __future__ import annotations

import json
from pathlib import Path

from clinidoc.cli import main
from clinidoc.dataset import load_dataset
from clinidoc.detectors import scan_all
from clinidoc.report import compute_verdict

DEMO = Path(__file__).resolve().parents[1] / "examples" / "demo_dataset"

PLANTED = {
    "structure.empty_text",
    "leakage.patient_across_splits",
    "duplicates.exact",
    "spans.out_of_range",
    "labels.thin_class",
    "split.class_missing_from_val",
    "phi.ssn",
    "temporal.test_before_train",
}


def test_demo_recalls_planted_defects() -> None:
    dataset = load_dataset(DEMO)
    findings = scan_all(dataset)
    found = {item.id for item in findings}
    missing = PLANTED - found
    assert not missing, f"missed planted defects: {missing}"
    assert compute_verdict(findings) == "blocked"
    ssn = next(item for item in findings if item.id == "phi.ssn")
    assert "078-05-1120" not in json.dumps(ssn.evidence)


def test_cli_audit_demo(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    code = main(["audit", str(DEMO), "--out", str(out), "--format", "json"])
    assert code == 2
    payload = json.loads((out / "findings.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == "blocked"
    found = {item["id"] for item in payload["findings"]}
    assert PLANTED <= found
    assert (out / "report.md").is_file()
    assert (out / "fix_plan.json").is_file()


def test_cli_scan_sarif() -> None:
    code = main(["scan", str(DEMO), "--format", "sarif"])
    assert code == 2


def test_cli_detectors() -> None:
    assert main(["detectors"]) == 0
