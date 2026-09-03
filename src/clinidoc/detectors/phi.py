from __future__ import annotations

import re

from clinidoc.dataset import LoadedDataset
from clinidoc.findings import PHI_EMAIL, PHI_IP, PHI_MRN, PHI_PHONE, PHI_SSN, CheckSpec, Finding, finding

# HIPAA Safe Harbor-style identifiers. Matches are reported with redacted snippets only.
PATTERNS: tuple[tuple[CheckSpec, re.Pattern[str]], ...] = (
    (PHI_SSN, re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    (
        PHI_PHONE,
        re.compile(
            r"(?<!\d)(?:\+1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]\d{4}(?!\d)"
        ),
    ),
    (PHI_EMAIL, re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    (PHI_MRN, re.compile(r"\b(?:MRN|mrn)[:\s#-]*[A-Z0-9]{5,}\b")),
    (PHI_IP, re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b")),
)

REDACTIONS = {
    "phi.ssn": "***SSN***",
    "phi.phone": "***PHONE***",
    "phi.email": "***EMAIL***",
    "phi.mrn": "***MRN***",
    "phi.ip": "***IP***",
}


def redact_snippet(text: str, start: int, end: int, token: str, window: int = 32) -> str:
    lo = max(0, start - window)
    hi = min(len(text), end + window)
    prefix = text[lo:start]
    suffix = text[end:hi]
    snippet = f"{prefix}{token}{suffix}".replace("\n", " ")
    if lo > 0:
        snippet = "…" + snippet
    if hi < len(text):
        snippet = snippet + "…"
    return snippet


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    for doc in dataset.documents:
        text = doc.text or ""
        if not text:
            continue
        for spec, pattern in PATTERNS:
            for match in pattern.finditer(text):
                token = REDACTIONS[spec.id]
                snippet = redact_snippet(text, match.start(), match.end(), token)
                findings.append(
                    finding(
                        spec,
                        f"Possible {spec.id.split('.', 1)[1].upper()} in document {doc.id}",
                        document_id=doc.id,
                        split=doc.split,
                        evidence={"snippet": snippet},
                    )
                )
    return findings
