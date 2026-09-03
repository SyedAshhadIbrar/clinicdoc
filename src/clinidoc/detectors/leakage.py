from __future__ import annotations

from collections import defaultdict

from clinidoc.dataset import Document, LoadedDataset
from clinidoc.detectors.duplicates import text_hash
from clinidoc.findings import LEAKAGE_DOCUMENT, LEAKAGE_PATIENT, Finding, finding

HELD_OUT = {"val", "test"}


def _cross_split(docs: list[Document]) -> bool:
    splits = {d.split for d in docs}
    return "train" in splits and bool(splits & HELD_OUT)


def scan(dataset: LoadedDataset) -> list[Finding]:
    findings: list[Finding] = []

    by_id: dict[str, list[Document]] = defaultdict(list)
    by_hash: dict[str, list[Document]] = defaultdict(list)
    by_patient: dict[str, list[Document]] = defaultdict(list)

    for doc in dataset.documents:
        by_id[doc.id].append(doc)
        if (doc.text or "").strip():
            by_hash[text_hash(doc.text)].append(doc)
        if doc.patient_id:
            by_patient[doc.patient_id].append(doc)

    reported_ids: set[str] = set()
    for doc_id, group in by_id.items():
        splits = {d.split for d in group}
        if len(splits) > 1:
            reported_ids.add(doc_id)
            findings.append(
                finding(
                    LEAKAGE_DOCUMENT,
                    f"Document id {doc_id} appears in splits {sorted(splits)}",
                    document_id=doc_id,
                    evidence={"document_ids": [doc_id], "splits": sorted(splits)},
                )
            )

    for digest, group in by_hash.items():
        splits = {d.split for d in group}
        if len(splits) <= 1:
            continue
        ids = [d.id for d in group]
        if all(i in reported_ids for i in ids):
            continue
        findings.append(
            finding(
                LEAKAGE_DOCUMENT,
                f"Identical text appears in splits {sorted(splits)} ({', '.join(ids)})",
                evidence={"document_ids": ids, "splits": sorted(splits), "hash": digest[:12]},
                count=len(group),
            )
        )

    for patient_id, group in by_patient.items():
        if not _cross_split(group):
            continue
        splits = sorted({d.split for d in group})
        ids = [d.id for d in group]
        findings.append(
            finding(
                LEAKAGE_PATIENT,
                f"patient_id {patient_id} appears in train and held-out splits {splits}",
                evidence={"patient_id": patient_id, "document_ids": ids, "splits": splits},
                count=len(group),
            )
        )
    return findings
