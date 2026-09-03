from __future__ import annotations

from clinidoc.dataset import LoadedDataset
from clinidoc.detectors import (
    duplicates,
    labels,
    leakage,
    phi,
    spans,
    split as split_mod,
    structure,
    temporal,
)
from clinidoc.findings import ALL_CHECKS, CheckSpec, Finding

DETECTORS = (
    structure,
    labels,
    spans,
    duplicates,
    leakage,
    split_mod,
    temporal,
    phi,
)


def scan_all(dataset: LoadedDataset) -> list[Finding]:
    findings = list(dataset.load_findings)
    for module in DETECTORS:
        findings.extend(module.scan(dataset))
    return findings


def list_checks() -> tuple[CheckSpec, ...]:
    return ALL_CHECKS
