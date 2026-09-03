from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SEVERITIES = ("critical", "major", "minor", "governance")
TRAINABILITY_GROUPS = (
    "structure",
    "labels",
    "spans",
    "duplicates",
    "leakage",
    "split",
    "temporal",
)
GROUPS = TRAINABILITY_GROUPS + ("phi",)

SEVERITY_RANK = {"critical": 3, "major": 2, "minor": 1, "governance": 0}


@dataclass(frozen=True)
class CheckSpec:
    id: str
    group: str
    default_severity: str
    description: str


@dataclass
class Finding:
    id: str
    severity: str
    group: str
    message: str
    document_id: str | None = None
    split: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    count: int = 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def finding(
    spec: CheckSpec,
    message: str,
    *,
    severity: str | None = None,
    document_id: str | None = None,
    split: str | None = None,
    evidence: dict[str, Any] | None = None,
    count: int = 1,
) -> Finding:
    return Finding(
        id=spec.id,
        severity=severity or spec.default_severity,
        group=spec.group,
        message=message,
        document_id=document_id,
        split=split,
        evidence=evidence or {},
        count=count,
    )


STRUCTURE_EMPTY_CORPUS = CheckSpec(
    "structure.empty_corpus",
    "structure",
    "critical",
    "No usable documents were loaded.",
)
STRUCTURE_NO_DOCUMENTS = CheckSpec(
    "structure.no_documents",
    "structure",
    "critical",
    "The path exists but produced zero documents.",
)
STRUCTURE_UNREADABLE = CheckSpec(
    "structure.unreadable_file",
    "structure",
    "critical",
    "A dataset file could not be opened or decoded.",
)
STRUCTURE_BAD_ROW = CheckSpec(
    "structure.bad_row",
    "structure",
    "major",
    "A row or line could not be parsed.",
)
STRUCTURE_EMPTY_TEXT = CheckSpec(
    "structure.empty_text",
    "structure",
    "major",
    "Document text is empty or whitespace-only.",
)
STRUCTURE_REPLACEMENT = CheckSpec(
    "structure.replacement_chars",
    "structure",
    "minor",
    "Replacement characters (U+FFFD) indicate a likely encoding error.",
)
STRUCTURE_MISSING_SPLIT = CheckSpec(
    "structure.missing_split_file",
    "structure",
    "major",
    "A split file declared in config is missing.",
)

LABELS_MISSING = CheckSpec(
    "labels.missing",
    "labels",
    "major",
    "Classification dataset documents are missing a label.",
)
LABELS_UNKNOWN = CheckSpec(
    "labels.unknown",
    "labels",
    "major",
    "A label is not in the configured allowed list.",
)
LABELS_CONFLICT = CheckSpec(
    "labels.conflict",
    "labels",
    "critical",
    "The same document id has conflicting labels.",
)
LABELS_THIN_CLASS = CheckSpec(
    "labels.thin_class",
    "labels",
    "major",
    "A class has too few examples to train or validate.",
)
LABELS_IMBALANCE = CheckSpec(
    "labels.imbalance",
    "labels",
    "minor",
    "Class distribution is severely imbalanced.",
)

SPANS_OUT_OF_RANGE = CheckSpec(
    "spans.out_of_range",
    "spans",
    "critical",
    "Entity offsets fall outside the document text.",
)
SPANS_INVALID_RANGE = CheckSpec(
    "spans.invalid_range",
    "spans",
    "critical",
    "Entity start is greater than or equal to end.",
)
SPANS_TEXT_MISMATCH = CheckSpec(
    "spans.text_mismatch",
    "spans",
    "major",
    "Stated span text does not match text[start:end].",
)
SPANS_INVALID_BIO = CheckSpec(
    "spans.invalid_bio",
    "spans",
    "critical",
    "BIO tag sequence is invalid (I- without B-, or type switch).",
)
SPANS_OVERLAP = CheckSpec(
    "spans.overlap_same_type",
    "spans",
    "major",
    "Two entities of the same type overlap.",
)

DUPLICATES_EXACT = CheckSpec(
    "duplicates.exact",
    "duplicates",
    "major",
    "Exact duplicate notes (identical text hash).",
)
DUPLICATES_NEAR = CheckSpec(
    "duplicates.near",
    "duplicates",
    "minor",
    "Near-duplicate notes (MinHash / shingle Jaccard).",
)

LEAKAGE_DOCUMENT = CheckSpec(
    "leakage.document_across_splits",
    "leakage",
    "critical",
    "The same document id or text appears in more than one split.",
)
LEAKAGE_PATIENT = CheckSpec(
    "leakage.patient_across_splits",
    "leakage",
    "critical",
    "The same patient_id appears in train and val/test.",
)

SPLIT_CLASS_MISSING = CheckSpec(
    "split.class_missing_from_val",
    "split",
    "major",
    "A class present in train is absent from val.",
)
SPLIT_UNUSABLE = CheckSpec(
    "split.unusable_size",
    "split",
    "major",
    "A split is empty or too small to be useful.",
)

TEMPORAL_TEST_BEFORE_TRAIN = CheckSpec(
    "temporal.test_before_train",
    "temporal",
    "major",
    "A val/test note is earlier than a train note for the same patient.",
)

PHI_SSN = CheckSpec("phi.ssn", "phi", "governance", "SSN-like identifier in note text.")
PHI_PHONE = CheckSpec("phi.phone", "phi", "governance", "Phone-number-like identifier in note text.")
PHI_EMAIL = CheckSpec("phi.email", "phi", "governance", "Email address in note text.")
PHI_MRN = CheckSpec("phi.mrn", "phi", "governance", "MRN-like token in note text.")
PHI_IP = CheckSpec("phi.ip", "phi", "governance", "IPv4 address in note text.")

ALL_CHECKS: tuple[CheckSpec, ...] = (
    STRUCTURE_EMPTY_CORPUS,
    STRUCTURE_NO_DOCUMENTS,
    STRUCTURE_UNREADABLE,
    STRUCTURE_BAD_ROW,
    STRUCTURE_EMPTY_TEXT,
    STRUCTURE_REPLACEMENT,
    STRUCTURE_MISSING_SPLIT,
    LABELS_MISSING,
    LABELS_UNKNOWN,
    LABELS_CONFLICT,
    LABELS_THIN_CLASS,
    LABELS_IMBALANCE,
    SPANS_OUT_OF_RANGE,
    SPANS_INVALID_RANGE,
    SPANS_TEXT_MISMATCH,
    SPANS_INVALID_BIO,
    SPANS_OVERLAP,
    DUPLICATES_EXACT,
    DUPLICATES_NEAR,
    LEAKAGE_DOCUMENT,
    LEAKAGE_PATIENT,
    SPLIT_CLASS_MISSING,
    SPLIT_UNUSABLE,
    TEMPORAL_TEST_BEFORE_TRAIN,
    PHI_SSN,
    PHI_PHONE,
    PHI_EMAIL,
    PHI_MRN,
    PHI_IP,
)

CHECKS_BY_ID = {c.id: c for c in ALL_CHECKS}

FIX_PLAN: dict[str, str] = {
    "structure.empty_corpus": "Fix unreadable files and empty notes until at least one document loads.",
    "structure.no_documents": "Confirm the path points at JSONL/CSV/BRAT/BIO files Clinicdoc can detect.",
    "structure.unreadable_file": "Repair encoding (UTF-8) or permissions on the listed file.",
    "structure.bad_row": "Fix or drop malformed JSON/CSV rows; re-run scan.",
    "structure.empty_text": "Remove or fill documents with empty text before training.",
    "structure.replacement_chars": "Re-export notes as UTF-8 without lossy conversion.",
    "structure.missing_split_file": "Add the missing split file or remove it from clinicdoc.yaml.",
    "labels.missing": "Label every classification document or drop unlabeled rows.",
    "labels.unknown": "Map unknown labels onto the allowed set, or extend the label list.",
    "labels.conflict": "Resolve conflicting labels for the same id; keep a single gold label.",
    "labels.thin_class": "Merge rare classes, collect more examples, or drop classes with <2 train rows.",
    "labels.imbalance": "Resample, reweight, or collect more minority-class notes.",
    "spans.out_of_range": "Clamp or recompute entity offsets so 0 <= start < end <= len(text).",
    "spans.invalid_range": "Drop or repair entities where start >= end.",
    "spans.text_mismatch": "Align span.text with the substring at [start:end], or drop the gold string.",
    "spans.invalid_bio": "Repair BIO sequences (I- must continue the same type as a preceding B-/I-).",
    "spans.overlap_same_type": "Merge or drop overlapping same-type entities.",
    "duplicates.exact": "Deduplicate identical notes; keep one copy per split policy.",
    "duplicates.near": "Review template-like near-duplicates; keep diversity or cap copies.",
    "leakage.document_across_splits": "Keep each document id/text in exactly one split.",
    "leakage.patient_across_splits": "Run `clinicdoc resplit PATH --by patient_id --out ./split`.",
    "split.class_missing_from_val": "Ensure every train class appears in val, or drop unused classes.",
    "split.unusable_size": "Increase val/test size or merge tiny splits.",
    "temporal.test_before_train": "Split patients so train notes are never later than that patient's val/test notes.",
    "phi.ssn": "De-identify SSN-like strings before any external sharing (Clinicdoc does not rewrite notes).",
    "phi.phone": "De-identify phone-like strings before any external sharing.",
    "phi.email": "De-identify email addresses before any external sharing.",
    "phi.mrn": "De-identify MRN-like tokens before any external sharing.",
    "phi.ip": "De-identify IP addresses before any external sharing.",
}


def is_trainability(finding: Finding) -> bool:
    return finding.group in TRAINABILITY_GROUPS
