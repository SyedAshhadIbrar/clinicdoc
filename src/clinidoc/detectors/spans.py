from __future__ import annotations

from clinidoc.dataset import LoadedDataset
from clinidoc.findings import (
    SPANS_INVALID_BIO,
    SPANS_INVALID_RANGE,
    SPANS_OUT_OF_RANGE,
    SPANS_OVERLAP,
    SPANS_TEXT_MISMATCH,
    Finding,
    finding,
)


def _bio_problems(tokens: list[tuple[str, str]]) -> list[str]:
    problems: list[str] = []
    prev_prefix, prev_label = "O", ""
    for i, (_tok, tag) in enumerate(tokens):
        prefix, _, label = tag.partition("-")
        prefix = prefix.upper()
        if tag.upper() in {"O", "0", ""}:
            prev_prefix, prev_label = "O", ""
            continue
        if prefix == "I":
            if prev_prefix not in {"B", "I"} or prev_label != label:
                problems.append(f"token {i}: {tag} after {prev_prefix}-{prev_label or 'O'}")
        prev_prefix, prev_label = prefix, label
    return problems


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []
    for doc in dataset.documents:
        if doc.tokens:
            for problem in _bio_problems(doc.tokens):
                findings.append(
                    finding(
                        SPANS_INVALID_BIO,
                        f"Document {doc.id} has invalid BIO sequence ({problem})",
                        document_id=doc.id,
                        split=doc.split,
                        evidence={"detail": problem},
                    )
                )
        entities = doc.entities or []
        n = len(doc.text or "")
        for ent in entities:
            if ent.start >= ent.end:
                findings.append(
                    finding(
                        SPANS_INVALID_RANGE,
                        f"Document {doc.id} has entity {ent.label} with start {ent.start} >= end {ent.end}",
                        document_id=doc.id,
                        split=doc.split,
                        evidence={"start": ent.start, "end": ent.end, "label": ent.label},
                    )
                )
                continue
            if ent.start < 0 or ent.end > n:
                findings.append(
                    finding(
                        SPANS_OUT_OF_RANGE,
                        f"Document {doc.id} has entity {ent.label} offsets [{ent.start}, {ent.end}] outside text length {n}",
                        document_id=doc.id,
                        split=doc.split,
                        evidence={"start": ent.start, "end": ent.end, "label": ent.label, "text_length": n},
                    )
                )
                continue
            if ent.text is not None:
                actual = (doc.text or "")[ent.start : ent.end]
                if actual != ent.text:
                    findings.append(
                        finding(
                            SPANS_TEXT_MISMATCH,
                            f"Document {doc.id} span {ent.label} text does not match offsets",
                            document_id=doc.id,
                            split=doc.split,
                            evidence={
                                "start": ent.start,
                                "end": ent.end,
                                "label": ent.label,
                                "expected": ent.text,
                                "actual": actual,
                            },
                        )
                    )
        same_type = sorted(entities, key=lambda e: (e.label, e.start, e.end))
        for i, left in enumerate(same_type):
            for right in same_type[i + 1 :]:
                if right.label != left.label:
                    break
                if left.start < right.end and right.start < left.end:
                    findings.append(
                        finding(
                            SPANS_OVERLAP,
                            f"Document {doc.id} has overlapping {left.label} entities",
                            document_id=doc.id,
                            split=doc.split,
                            evidence={
                                "label": left.label,
                                "a": [left.start, left.end],
                                "b": [right.start, right.end],
                            },
                        )
                    )
    return findings
