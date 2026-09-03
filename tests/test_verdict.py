from __future__ import annotations

from clinidoc.findings import Finding
from clinidoc.report import compute_verdict, exit_code, findings_to_sarif


def _f(severity: str, group: str, fid: str = "x") -> Finding:
    return Finding(id=fid, severity=severity, group=group, message="m")


def test_verdict_blocked_on_critical() -> None:
    assert compute_verdict([_f("critical", "leakage")]) == "blocked"


def test_verdict_caution_on_major_minor() -> None:
    assert compute_verdict([_f("major", "labels")]) == "caution"
    assert compute_verdict([_f("minor", "duplicates")]) == "caution"


def test_verdict_pass_on_phi_only() -> None:
    assert compute_verdict([_f("governance", "phi", "phi.ssn")]) == "pass"


def test_verdict_phi_blocks_when_requested() -> None:
    assert compute_verdict([_f("governance", "phi", "phi.ssn")], fail_on_phi=True) == "blocked"


def test_exit_codes() -> None:
    blocked = [_f("critical", "leakage")]
    caution = [_f("major", "labels")]
    assert exit_code("blocked", blocked, fail_on="critical") == 2
    assert exit_code("caution", caution, fail_on="critical") == 0
    assert exit_code("caution", caution, fail_on="major") == 1
    assert exit_code("pass", [], fail_on="critical") == 0


def test_sarif_shape() -> None:
    payload = findings_to_sarif([_f("critical", "leakage", "leakage.patient_across_splits")])
    assert payload["version"] == "2.1.0"
    assert payload["runs"][0]["results"][0]["ruleId"] == "leakage.patient_across_splits"
    assert payload["runs"][0]["results"][0]["level"] == "error"
